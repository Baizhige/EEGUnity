import ast
import copy
import json
import os
import warnings

import mne
import numpy as np
import pandas as pd
from typing import Callable, Union, Tuple, List, Dict, Optional

from eegunity.module_eeg_batch.eeg_scores import calculate_eeg_quality_scores
from eegunity.module_eeg_parser.eeg_parser import get_data_row, format_channel_names, extract_events, infer_channel_unit
from eegunity.share_attributes import UDatasetSharedAttributes

current_dir = os.path.dirname(__file__)
json_file_path = os.path.join(current_dir, '..', 'module_eeg_parser', 'combined_montage.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)
STANDARD_EEG_CHANNELS = list(data.keys())


class EEGBatch(UDatasetSharedAttributes):
    '''
    This is a key module of UnifiedDataset class, with focus on batch processing.
    This EEGBatch class has the same attributes as the UnifiedDataset class.
    '''
    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr
        self.main_instance = main_instance
    def batch_process(self,
                      con_func: Callable,
                      app_func: Callable,
                      is_patch: bool,
                      result_type: Union[str, None] = None):
        r"""Process each row of the given dataframe locator based on conditions specified
        in `con_func` and apply `app_func` accordingly. This function handles both list
        and dataframe return types, ensuring the result aligns with the original locator's
        rows based on the `is_patch` flag.

        Parameters
        ----------
        con_func : Callable
            A function that takes a row of locator and returns `True` or `False` to determine
            if `app_func` should be applied to that row.
        app_func : Callable
            A function that processes a row of locator and returns the result.
        is_patch : bool
            If `True`, the returned list length or dataframe rows will match the locator's
            row count, using placeholder elements as needed.
        result_type : {'series', 'value', None}, optional
            Specifies the expected return type of `app_func` results. Can be "series", "value",
            or `None` (case insensitive). Defaults to `None`.

        Returns
        -------
        Union[None, list, pd.DataFrame]
            The processed results, either as a list or dataframe, depending on `result_type`
            and `app_func` return type and consistency. Returns `None` if `result_type` is `None`.

        Raises
        ------
        ValueError
            If `result_type` is not one of the expected values.

        Notes
        -----
        Ensure that `con_func` and `app_func` are compatible with the structure of the locator.
        If using `is_patch`, consider the implications on the data integrity.

        Examples
        --------
        >>> results = batch_process(locator, condition_function, application_function, True, 'series')
        >>> print(results)
        [expected result format]
        """

        if result_type is not None:
            result_type = result_type.lower()
            if result_type not in ["series", "value", None]:
                raise ValueError("Invalid result_type. Must be 'series', 'value', or None.")

        results = []
        for index, row in self.get_shared_attr()['locator'].iterrows():
            if con_func(row):
                result = app_func(row)
            else:
                if is_patch and result_type == "series":
                    result = row
                else:
                    result = None

            results.append(result)

        if result_type == "series":
            # Check if all elements in results are None
            if all(res is None for res in results):
                warnings.warn("The file list are empty. Returning an empty Locator. Please check the con_func() and app_func().")
                # Create an empty DataFrame with the same columns as `locator`
                empty_df = pd.DataFrame(columns=self.get_shared_attr()['locator'].columns)
                return empty_df
            else:
                # Combine results into a DataFrame if app_func returns Series
                combined_results = pd.concat([res for res in results if res is not None], axis=1).T
                combined_results.reset_index(drop=True, inplace=True)
                return combined_results
        elif result_type == "value":
            # Collect results into a list if app_func returns values
            if is_patch:
                return results
            else:
                return [res for res in results if res is not None]
        else:
            return None

    def set_column(self, col_name: str, value: list):
        r"""Set the specified column in the locator with the given list of values.

        Parameters
        ----------
        col_name : str
            The name of the column to be set.
        value : list
            The list of values to set in the column. Its length must match the number of rows
            in the dataframe.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the length of the `value` list does not match the number of rows in the dataframe.
        TypeError
            If the input types are not as expected (e.g., `col_name` is not a string or `value` is not a list).

        Notes
        -----
        Ensure that the provided `value` list contains valid entries for the specified column type.

        Examples
        --------
        >>> locator.set_column('column_name', [1, 2, 3])
        >>> print(locator.dataframe['column_name'])
        [1, 2, 3]
        """
        # Check if locator is a DataFrame
        if not isinstance(self.get_shared_attr()['locator'], pd.DataFrame):
            raise TypeError("locator must be a pandas DataFrame")

        # Check if col_name is a string
        if not isinstance(col_name, str):
            raise TypeError("col_name must be a string")

        # Check if value is a list
        if not isinstance(value, list):
            raise TypeError("value must be a list")

        # Check if the length of value matches the number of rows in locator
        if len(value) != len(self.get_shared_attr()['locator']):
            raise ValueError("Length of value list must match the number of rows in the dataframe")

        # Set the column with the provided values
        self.get_shared_attr()['locator'][col_name] = value

    def sample_filter(
            self,
            channel_number: Union[Tuple[int, int], List[int], None] = None,
            sampling_rate: Union[Tuple[float, float], List[float], None] = None,
            duration: Union[Tuple[float, float], List[float], None] = None,
            completeness_check: Union[str, None] = None,
            domain_tag: Union[str, None] = None,
            file_type: Union[str, None] = None
    ) -> None:
        r"""Filters the 'locator' dataframe based on the given criteria. This function is typically
        used to select the data file according to specified requirements. For advanced filtering,
        refer to the `batch_process()` method.

        Parameters
        ----------
        channel_number : Union[Tuple[int, int], List[int], None], optional
            A tuple or list with (min, max) values to filter the "Number of Channels" column.
            If `None`, this criterion is ignored. Defaults to `None`.
        sampling_rate : Union[Tuple[float, float], List[float], None], optional
            A tuple or list with (min, max) values to filter the "Sampling Rate" column.
            If `None`, this criterion is ignored. Defaults to `None`.
        duration : Union[Tuple[float, float], List[float], None], optional
            A tuple or list with (min, max) values to filter the "Duration" column.
            If `None`, this criterion is ignored. Defaults to `None`.
        completeness_check : str, optional
            A string that can be 'Completed', 'Unavailable', or 'Acceptable' to filter the
            "Completeness Check" column. The check is case-insensitive. If `None`, this criterion
            is ignored. Defaults to `None`.
        domain_tag : str, optional
            A string to filter the "Domain Tag" column. If `None`, this criterion is ignored.
            Defaults to `None`.
        file_type : str, optional
            A string to filter the "File Type" column. If `None`, this criterion is ignored.
            Defaults to `None`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any of the input parameters are not in the expected format (e.g., invalid tuples or
            strings).

        Notes
        -----
        This method modifies the 'locator' dataframe in place based on the provided filters.

        Examples
        --------
        >>> locator.sample_filter(channel_number=(2, 8), completeness_check='Completed')
        >>> print(locator.filtered_data)
        [expected filtered results]
        """

        def con_func(row):
            """
            Condition function to determine if a row meets the filtering criteria.

            Parameters:
            row (pd.Series): A row from the 'locator' dataframe.

            Returns:
            bool: True if the row meets all the specified criteria, False otherwise.
            """
            if channel_number:
                min_channels, max_channels = channel_number
                if not (min_channels <= float(row["Number of Channels"]) <= max_channels):
                    return False
            if sampling_rate:
                min_rate, max_rate = sampling_rate
                if not (min_rate <= float(row["Sampling Rate"]) <= max_rate):
                    return False
            if duration:
                min_duration, max_duration = duration
                if not (min_duration <= float(row["Duration"]) <= max_duration):
                    return False
            if completeness_check:
                if row["Completeness Check"].strip().lower() != completeness_check.strip().lower():
                    return False
            if domain_tag:
                if row["Domain Tag"] != domain_tag:
                    return False
            if file_type:
                if row["File Type"] != file_type:
                    return False
            return True

        def app_func(row):
            return row

        # Process the dataframe
        filtered_locator = self.batch_process(con_func, app_func, is_patch=False, result_type='series')

        # Update the locator in shared attributes
        self.set_shared_attr({'locator': filtered_locator})

    def save_as_other(self,
                      output_path: str,
                      domain_tag: Union[str, None] = None,
                      format: str = 'fif',
                      preserve_events: bool = True):  # New parameter to control event preservation
        r"""Save data in the specified format ('fif' or 'csv') to the given output path.

        Parameters
        ----------
        output_path : str
            The directory path where the converted files will be saved. If the path does not exist,
            a `FileNotFoundError` is raised.
        domain_tag : str, optional
            Optional filter to save only the files with a matching 'Domain Tag'. If `None`, all files
            are processed.
        format : str, optional
            The format to save the data in. Supported formats are 'fif' and 'csv'. If an unsupported
            format is provided, a `ValueError` is raised. Defaults to 'fif'.
        preserve_events : bool, optional
            If `True`, event markers will be included in the CSV file, and metadata will be adjusted.
            Defaults to `True`.

        Returns
        -------
        instance of the same class
            A copied instance of the class with updated file paths and formats after the batch
            processing is complete.

        Raises
        ------
        FileNotFoundError
            If the output path does not exist.
        ValueError
            If the `format` is not 'fif' or 'csv'.

        Notes
        -----
        Ensure that the `output_path` is accessible and has the necessary write permissions.

        Examples
        --------
        >>> new_locator = locator.save_as_other('/path/to/output', domain_tag='example', format='csv')
        >>> print(new_locator.file_paths)
        [expected updated file paths]
        """
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output path does not exist: {output_path}")

        if format not in ['fif', 'csv']:
            raise ValueError(f"Unsupported format: {format}. Currently, only 'fif' and 'csv' are supported.")

        def con_func(row):
            return domain_tag is None or row['Domain Tag'] == domain_tag

        def app_func(row):
            raw = get_data_row(row, is_set_channel_type=False)
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]

            if format == 'fif':
                new_file_path = os.path.join(output_path, f"{file_name}.fif")
                raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path
                row['File Type'] = "standard_data"

            elif format == 'csv':
                new_file_path = os.path.join(output_path, f"{file_name}.csv")

                # Extract data and channel names from Raw object
                data, times = raw.get_data(return_times=True)
                channel_names = raw.info['ch_names']

                # Create a DataFrame with 'date' column and channel data
                df = pd.DataFrame(data.T, columns=channel_names)
                df.insert(0, 'timestamp', times)  # Add 'date' as the first column

                if preserve_events:
                    # Extract events from raw data
                    events, event_id = extract_events(raw)

                    # Add a 'marker' column for event codes
                    df['marker'] = np.nan
                    for event in events:
                        onset_sample = event[0]
                        event_code = event[2]
                        closest_time_idx = np.argmin(np.abs(times - raw.times[onset_sample]))
                        df.at[closest_time_idx, 'marker'] = event_code

                    # Adjust metadata in 'row'
                    row['Number of Channels'] = str(int(row['Number of Channels']) + 1)
                    row['Channel Names'] += ', marker'
                    row['Channel Names'] = 'timestamp, ' + row['Channel Names']
                    # Adjust 'Data Shape'
                    data_shape = eval(row['Data Shape'])
                    smaller_shape, larger_shape = min(data_shape), max(data_shape)
                    updated_shape = f"({smaller_shape + 1}, {larger_shape})" if smaller_shape == data_shape[
                        0] else f"({larger_shape}, {smaller_shape + 1})"
                    row['Data Shape'] = updated_shape

                # Save DataFrame to CSV
                df.to_csv(new_file_path, index=False)
                row['File Path'] = new_file_path
                row['File Type'] = "csvData"

            return row

        copied_instance = copy.deepcopy(self)
        new_locator = self.batch_process(con_func, app_func, is_patch=False, result_type='series')
        copied_instance.set_shared_attr({'locator': new_locator})
        return copied_instance

    def process_mean_std(self, domain_mean: bool = True) -> None:
        r"""Process the mean and standard deviation for EEG data across different channels and optionally
        compute domain-level statistics.

        This function calculates the mean and standard deviation for all EEG channels, both combined and individually.
        It can also aggregate the results by domain if `domain_mean` is set to `True`.

        Parameters
        ----------
        domain_mean : bool, optional
            If `True` (default), the function aggregates the results by domain tags. Each domain contains the
            mean and standard deviation across all related EEG channels. If `False`, the function calculates
            and stores individual mean and standard deviation for each EEG recording.

        Returns
        -------
        None
            The function updates the instance by setting the "MEAN STD" column with the calculated mean and
            standard deviation values. If `domain_mean` is `True`, it computes domain-aggregated statistics;
            otherwise, it stores per-channel results.

        Raises
        ------
        ValueError
            If inconsistent channel names or numbers are found within a domain when `domain_mean` is `True`.

        Notes
        -----
        Ensure that the EEG data is properly formatted and that all necessary channels are present before calling
        this method.

        Examples
        --------
        >>> locator.process_mean_std(domain_mean=True)
        >>> print(locator.dataframe['MEAN STD'])
        [expected mean and standard deviation results]
        """
        def get_mean_std(data: mne.io.Raw):
            """
            Calculate mean and standard deviation for each channel and all channels in the given EEG data.

            Parameters:
            data (mne.io.Raw): The raw EEG data.

            Returns:
            dict: A dictionary containing the mean and standard deviation for all channels combined
                  and for each individual channel.
            """
            # Get data from Raw object
            data_array = data.get_data()

            # Get EEG channel indices
            eeg_channel_indices = mne.pick_types(data.info, eeg=True, meg=False, stim=False, eog=False)

            # Calculate mean and std for all EEG channels combined
            eeg_data = data_array[eeg_channel_indices, :]
            all_mean = np.mean(eeg_data)
            all_std = np.std(eeg_data)

            # Initialize result dictionary
            result = {"all_eeg": (all_mean, all_std)}

            # Calculate mean and std for each channel
            for i, ch_name in enumerate(data.ch_names):
                ch_mean = np.mean(data_array[i, :])
                ch_std = np.std(data_array[i, :])
                result[ch_name] = (ch_mean, ch_std)

            return result

        def con_func(row):
            return True

        def app_func(row):
            data = get_data_row(row)
            mean_std = get_mean_std(data)
            return mean_std

        results = self.batch_process(con_func, app_func, is_patch=False, result_type="value")

        if domain_mean:
            domain_results = {}
            for index, row in self.get_shared_attr()['locator'].iterrows():
                domain_tag = row['Domain Tag']
                if domain_tag not in domain_results:
                    domain_results[domain_tag] = []
                domain_results[domain_tag].append(results[index])

            # Check for channel consistency within each domain
            for domain_tag, res_list in domain_results.items():
                all_keys_sets = [set(res.keys()) - {'all'} for res in res_list]  # Exclude the 'all' key
                if len(set(map(frozenset, all_keys_sets))) != 1:
                    raise ValueError(
                        f"Inconsistent channel names in domain '{domain_tag}', please try domain_mean=False or check the locator manually and make sure all channel names are the same within a domain.")
                num_channels = [len(keys) for keys in all_keys_sets]
                if len(set(num_channels)) != 1:
                    raise ValueError(
                        f"Inconsistent number of channels in domain '{domain_tag}', please try domain_mean=False or check the locator manually and make sure all channel names are the same within a domain.")

            domain_mean_std = {}
            for domain_tag, res_list in domain_results.items():
                all_keys = res_list[0].keys()
                mean_std_dict = {}
                for key in all_keys:
                    mean_sum = np.mean([res[key][0] for res in res_list])
                    std_sum = np.mean([res[key][1] for res in res_list])
                    mean_std_dict[key] = (mean_sum, std_sum)
                domain_mean_std[domain_tag] = mean_std_dict

            new_results = []
            for index, row in self.get_shared_attr()['locator'].iterrows():
                domain_tag = row['Domain Tag']
                new_results.append(domain_mean_std[domain_tag])
            self.set_column("MEAN STD", new_results)
        else:
            self.set_column("MEAN STD", results)

    def format_channel_names(self):
        r"""Format channel names in the dataset and update the 'Channel Names' column.

        This function processes each row in the dataset, checks the 'Channel Names'
        column, and applies a formatting function to standardize the channel names.
        A cache is used to avoid redundant formatting operations for channel names
        that have already been processed. The function utilizes the `batch_process` method
        to apply the formatting to each row, and the updated channel names are then saved
        back to the 'Channel Names' column.

        Raises
        ------
        KeyError
            If the 'Channel Names' column is missing from the dataset.

        Returns
        -------
        None
            The function modifies the dataset in place by updating the
            'Channel Names' column.

        Notes
        -----
        Ensure that the dataset is properly loaded and contains the 'Channel Names' column
        before calling this method.

        Examples
        --------
        >>> locator.format_channel_names()
        >>> print(locator.dataframe['Channel Names'])
        [expected formatted channel names]
        """

        cache = {}

        def con_func(row):
            return True

        def app_func(row):
            channel_name = row['Channel Names']
            if channel_name in cache:
                return cache[channel_name]
            else:
                formatted_channel_name = format_channel_names(channel_name)
                cache[channel_name] = formatted_channel_name
                return formatted_channel_name

        results = self.batch_process(con_func, app_func, is_patch=False, result_type="value")

        self.set_column("Channel Names", results)

    def filter(self,
               output_path: str,
               filter_type: str = 'bandpass',
               l_freq: float = None,
               h_freq: float = None,
               notch_freq: float = None,
               auto_adjust_h_freq: bool = True,
               picks: str = 'all',
               miss_bad_data: bool = False,
               **kwargs) -> None:
        r"""Apply filtering to the data, supporting low-pass, high-pass, band-pass, and notch filters.

        Parameters
        ----------
        output_path : str
            Path to save the filtered file.
        filter_type : {'lowpass', 'highpass', 'bandpass', 'notch'}, optional
            Type of filter to apply. Defaults to 'bandpass'.
        l_freq : float, optional
            Low cutoff frequency for the filter (used in high-pass or low-frequency band-pass filters). Defaults to `None`.
        h_freq : float, optional
            High cutoff frequency for the filter (used in low-pass or high-frequency band-pass filters). Defaults to `None`.
        notch_freq : float, optional
            Frequency for the notch filter. Defaults to `None`.
        auto_adjust_h_freq : bool, optional
            Whether to automatically adjust the high cutoff frequency to fit the Nyquist frequency. Defaults to `True`.
        picks : str, optional
            Channels to be used for filtering. Defaults to 'all'.
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs. Defaults to `False`.
        **kwargs : dict
            Additional keyword arguments for `mne_raw.filter()` and `mne_raw.notch_filter()`.

        Returns
        -------
        None
            The function modifies the dataset in place.

        Raises
        ------
        ValueError
            If an invalid `filter_type` is specified or if cutoff frequencies are inconsistent.

        Notes
        -----
        Ensure that the output path is accessible and that all necessary channels are specified for filtering.

        Examples
        --------
        >>> locator.filter('/path/to/save/filtered_file.fif', filter_type='bandpass', l_freq=1.0, h_freq=40.0)
        >>> print("Filtering complete.")
        """

        def con_func(row):
            return True

        def app_func(row, l_freq, h_freq, notch_freq, filter_type, output_path, auto_adjust_h_freq, picks,
                     **kwargs):
            try:
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_filter.fif")
                mne_raw = get_data_row(row)

                # get sampling rate
                sfreq = mne_raw.info['sfreq']
                nyquist_freq = sfreq / 2.0

                # adjust high pass frequency based on nyquist frequency
                if h_freq is not None and h_freq >= nyquist_freq:
                    if auto_adjust_h_freq:
                        warnings.warn(
                            f"High-pass frequency ({h_freq} Hz) is greater than or equal to Nyquist frequency ({nyquist_freq} Hz). Adjusting h_freq to {nyquist_freq - 1} Hz.")
                        h_freq = nyquist_freq - 1
                    else:
                        raise ValueError(
                            f"High-pass frequency must be less than Nyquist frequency ({nyquist_freq} Hz).")

                # add l_freq and h_freq to kwargs
                if filter_type in ['lowpass', 'highpass', 'bandpass']:
                    kwargs['l_freq'] = l_freq
                    kwargs['h_freq'] = h_freq
                    kwargs['fir_design'] = 'firwin'

                # apply the appropriate filter based on filter_type
                if filter_type in ['lowpass', 'highpass', 'bandpass']:
                    mne_raw.filter(picks=picks, **kwargs)
                elif filter_type == 'notch':
                    if notch_freq is not None:
                        mne_raw.notch_filter(freqs=notch_freq, picks=picks, **kwargs)
                    else:
                        raise ValueError("notch_freq must be specified for notch filter.")
                else:
                    raise ValueError("Invalid filter_type. Must be 'lowpass', 'highpass', 'bandpass', or 'notch'.")

                # save filtered data
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path

                return new_file_path
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return ""  # Return an empty path to indicate failure
                else:
                    raise

        new_path_list = self.batch_process(con_func,
                                           lambda row: app_func(row, l_freq, h_freq, notch_freq, filter_type,
                                                                output_path, auto_adjust_h_freq, picks,
                                                                **kwargs),
                                           is_patch=False,
                                           result_type='value')

        self.set_column("File Path", new_path_list)
        locator_df = self.get_shared_attr()['locator']
        locator_df = locator_df[locator_df['File Path'] != ""]
        self.get_shared_attr()['locator'] = locator_df

    def ica(self, output_path: str, miss_bad_data: bool = False, **kwargs: Dict):
        r"""Apply ICA (Independent Component Analysis) to the specified file in the dataset.

        This method applies ICA to clean the EEG data using parameters passed through **kwargs.
        Please refer to the official documentation for `mne.preprocessing.ICA` and `ica.fit()` for the
        complete list of available parameters.

        Documentation links:
        - `mne.preprocessing.ICA`: https://mne.tools/stable/generated/mne.preprocessing.ICA.html

        Parameters
        ----------
        output_path : str
            Path to save the processed file after applying ICA.
        miss_bad_data : bool, optional
            Whether to skip bad data files and continue processing the next one. Defaults to `False`.
        **kwargs : dict
            Additional parameters passed to `mne.preprocessing.ICA`, `ica.fit()`, and other MNE functions.
            This includes `picks`, `n_components`, `method`, `random_state`, etc.

        Returns
        -------
        None
            Updates the file path in the dataset locator after ICA is applied.

        Raises
        ------
        ValueError
            If the output path is invalid or if the specified parameters in `kwargs` are inconsistent.

        Notes
        -----
        Ensure that the input data is properly formatted and that all necessary parameters are specified
        before calling this method.

        Examples
        --------
        >>> locator.ica('/path/to/save/ica_processed_file.fif', n_components=20)
        >>> print("ICA processing complete.")
        """

        def con_func(row):
            return True

        def app_func(row, output_path: str):
            try:
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_ica_cleaned_raw.fif")
                mne_raw = get_data_row(row)

                # Initialize ICA with additional kwargs
                ica = mne.preprocessing.ICA(**kwargs)

                # Fit ICA with kwargs
                ica.fit(mne_raw, **kwargs)

                # Apply ICA to raw data with kwargs
                ica.apply(mne_raw, **kwargs)

                # Save file and update path in locator
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path

                return new_file_path
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return ""  # Return empty path to indicate failure
                else:
                    raise

        new_path_list = self.batch_process(con_func,
                                           lambda row: app_func(row, output_path),
                                           is_patch=False,
                                           result_type='value')

        # Update locator
        self.set_column("File Path", new_path_list)
        locator_df = self.get_shared_attr()['locator']
        locator_df = locator_df[locator_df['File Path'] != ""]
        self.get_shared_attr()['locator'] = locator_df

    def resample(self, output_path: str, miss_bad_data: bool = False, **kwargs) -> None:
        r"""Resample the data using MNE's resampling functionality and save the processed data.

        Parameters
        ----------
        output_path : str
            The path where the resampled file will be saved.
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs. Defaults to `False`.
        **kwargs : dict
            Additional parameters to be passed to the `mne_raw.resample()` function. For detailed information about
            these parameters, refer to the MNE documentation:
            https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample

        Returns
        -------
        None
            The function modifies the dataset in place by saving the resampled data.

        Raises
        ------
        Exception
            If an error occurs during resampling and `miss_bad_data` is set to `False`, the error will be raised.

        Notes
        -----
        Ensure that the output path is accessible and that the input data is properly formatted before calling this method.

        Examples
        --------
        >>> locator.resample('/path/to/save/resampled_file.fif', new_sfreq=256)
        >>> print("Resampling complete.")
        """

        def con_func(row) -> bool:
            return True

        def app_func(row, output_path: str) -> dict:
            try:
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_resampled_raw.fif")

                mne_raw = get_data_row(row)

                # Resample using the provided kwargs
                mne_raw.resample(**kwargs)

                # Save resampled data and update locator
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path
                row['Duration'] = str(
                    float(row['Sampling Rate'].strip()) * float(row['Duration'].strip()) / int(mne_raw.info['sfreq']))
                row['Sampling Rate'] = str(int(mne_raw.info['sfreq']))

                dimensions = row['Data Shape'].strip('()').split(',')
                dimensions = [int(dim.strip()) for dim in dimensions]
                dimensions[dimensions.index(max(dimensions))] = int(
                    float(row['Duration'].strip()) * float(mne_raw.info['sfreq']))
                row['Data Shape'] = f"({dimensions[0]}, {dimensions[1]})"
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return None to indicate failure
                else:
                    raise

        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row, output_path),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def align_channel(self, output_path: str, channel_order: list, min_num_channels: int = 1,
                      miss_bad_data: bool = False, **kwargs: Dict):
        r"""Adjust the channel order and perform interpolation on the data.

        This method realigns the EEG data channels based on the provided `channel_order`. It utilizes
        `get_data_row()` for retrieving the data. Additional parameters can be passed to `get_data_row()`
        via `**kwargs`. For more information on available options, refer to the
        :func:`get_data_row` function in this documentation.

        Parameters
        ----------
        output_path : str
            The path where the adjusted file will be saved.
        channel_order : list
            The desired order of channels, provided as a list.
        min_num_channels : int, optional
            The minimum number of channels required for alignment. Defaults to 1.
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs.
            Defaults to `False`.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to :func:`get_data_row` for data fetching.
            This allows fine-tuning the data retrieval process.

        Returns
        -------
        None
            The function modifies the dataset in place by saving the adjusted data.

        Raises
        ------
        ValueError
            If any invalid channels are found in the provided `channel_order` or if the number of
            matching channels is below `min_num_channels`.

        Notes
        -----
        Ensure that the output path is accessible and that the provided channel order is valid before calling this method.

        Examples
        --------
        >>> locator.align_channel('/path/to/save/aligned_file.fif', channel_order=['C3', 'C4', 'O1'], min_num_channels=3)
        >>> print("Channel alignment complete.")
        """

        invalid_channels = [ch for ch in channel_order if ch not in STANDARD_EEG_CHANNELS]
        if invalid_channels:
            raise ValueError(
                f"Invalid channels found: {invalid_channels}. All specified channels must be in the standard channel list.")

        def con_func(row):
            return True

        def app_func(row, channel_order, output_path, min_num_channels):
            try:
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_aligned.fif")

                # Fetch the data using get_data_row and **kwargs
                mne_raw = get_data_row(row, **kwargs)

                existing_channels = mne_raw.ch_names
                matched_channels = [ch for ch in channel_order if ch in existing_channels]

                if len(matched_channels) < min_num_channels:
                    if miss_bad_data:
                        print(f"File {row['File Path']} skipped due to insufficient matching channels.")
                        return None  # Return an empty path to indicate failure
                    else:
                        raise ValueError(
                            f"File {row['File Path']} has insufficient matching channels. Required: {min_num_channels}, Found: {len(matched_channels)}")

                # Pick the matched channels and ensure the correct order
                mne_raw.pick_channels(matched_channels, ordered=True)

                if len(matched_channels) < len(channel_order):
                    missing_channels = [ch for ch in channel_order if ch not in existing_channels]

                    # Preload the data before adding channels
                    mne_raw.load_data()

                    # Create missing channels and add them
                    missing_info = mne.create_info(missing_channels, sfreq=mne_raw.info['sfreq'], ch_types='eeg')
                    missing_raw = mne.io.RawArray(np.zeros((len(missing_channels), len(mne_raw.times))), missing_info)
                    mne_raw.add_channels([missing_raw])

                    # Interpolate missing channels
                    mne_raw.interpolate_bads(reset_bads=False, origin='auto')

                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path
                row['Channel Names'] = ', '.join(mne_raw.info['ch_names'])
                row['Number of Channels'] = str(len(channel_order))
                dimensions = row['Data Shape'].strip('()').split(',')
                dimensions = [int(dim.strip()) for dim in dimensions]
                dimensions[dimensions.index(min(dimensions))] = len(channel_order)
                row['Data Shape'] = f"({dimensions[0]}, {dimensions[1]})"
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None
                else:
                    raise

        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row, channel_order, output_path, min_num_channels),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def normalize(self, output_path: str, norm_type: str = 'sample-wise', miss_bad_data: bool = False,
                  domain_mean: bool = True, **kwargs):
        r"""Normalize the data.

        This method normalizes the EEG data based on the specified normalization type. It can either
        perform sample-wise normalization or aggregate by domain mean, depending on the provided parameters.

        Parameters
        ----------
        output_path : str
            The path where the normalized file will be saved.
        norm_type : {'sample-wise', 'domain-wise'}, optional
            The type of normalization to apply. Defaults to 'sample-wise'.
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs.
            Defaults to `False`.
        domain_mean : bool, optional
            If True (default), the function aggregates the results by domain tags. Each domain contains the
            mean and standard deviation across all related EEG channels. If False, the function calculates
            and stores individual mean and standard deviation for each EEG recording.
        **kwargs : dict, optional
            Additional keyword arguments passed to :func:`get_data_row`. This allows users to pass extra parameters
            required by the `get_data_row` function seamlessly. For details on the parameters, refer to the
            :func:`get_data_row()` function in this documentation.

        Returns
        -------
        None
            The function modifies the dataset in place by saving the normalized data.

        Raises
        ------
        ValueError
            If the specified normalization type is invalid or if there are issues with the input data.

        Notes
        -----
        Ensure that the output path is accessible and that the input data is properly formatted before calling this method.

        Examples
        --------
        >>> locator.normalize('/path/to/save/normalized_file.fif', norm_type='domain-wise', domain_mean=True)
        >>> print("Normalization complete.")
        """

        # Check if 'MEAN STD' column exists, if not process mean and std
        locator = self.get_shared_attr()['locator']
        if 'MEAN STD' not in locator.columns:
            self.process_mean_std(domain_mean=domain_mean)

        def con_func(row):
            return True

        def app_func(row, norm_type, output_path):
            try:
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_normed.fif")
                mne_raw = get_data_row(row, norm_type=norm_type, **kwargs)
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path
                row['File Type'] = "standard_data"
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return None to indicate failure
                else:
                    raise

        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row, norm_type, output_path),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def epoch_for_pretraining(self,
                              output_path: str,
                              seg_sec: float,
                              resample: Optional[int] = None,
                              overlap: float = 0.0,
                              exclude_bad: bool = True,
                              baseline: Tuple[Optional[float], float] = (None, 0),
                              miss_bad_data: bool = False,
                              **kwargs) -> None:
        r"""Processes data by creating epochs for pretraining from raw EEG data, applying optional resampling and event
        segmentation.

        Parameters
        ----------
        output_path : str
            Path to save the preprocessed epoch data in .npy format.
        seg_sec : float
            Segment length in seconds for each epoch.
        resample : Optional[int], optional
            New sampling rate. If specified, raw data will be resampled.
        overlap : float, optional
            Fraction of overlap between consecutive segments (0.0 means no overlap).
        exclude_bad : bool, optional
            If True, drops epochs marked as bad.
        baseline : Tuple[Optional[float], float], optional
            Baseline correction period, represented as a tuple (start, end). Default is (None, 0).
        miss_bad_data : bool, optional
            If True, skips files with errors instead of raising an exception.
        **kwargs : dict, optional
            Additional keyword arguments passed to `mne.Epochs()` and `raw_data.resample()`
            for further filtering options. Refer to the respective functions' documentation for details.

        Returns
        -------
        None
            The function modifies the dataset in place by saving the processed epoch data.

        Raises
        ------
        ValueError
            If the segment length is invalid or if any specified parameters are inconsistent.

        Notes
        -----
        Ensure that the output path is accessible and that the input data is properly formatted before calling this method.

        Examples
        --------
        >>> locator.epoch_for_pretraining('/path/to/save/epochs.npy', seg_sec=2.0, resample=256)
        >>> print("Epoch creation for pretraining complete.")
        """

        def con_func(row) -> bool:
            """Condition function to always process the row."""
            return True

        def app_func(row: Dict,
                     output_path: str,
                     seg_sec: float,
                     resample: Optional[int] = None,
                     overlap: float = 0.0,
                     exclude_bad: bool = True,
                     baseline: Tuple[Optional[float], float] = None,
                     **kwargs) -> Optional[None]:
            """
            Applies the epoch processing to a single row of data.

            :param row: A dictionary representing a data row, including 'File Path' for raw EEG file.
            :param output_path: Path to save the epoch data.
            :param seg_sec: Length of each epoch in seconds.
            :param resample: Optional resampling rate.
            :param overlap: Overlap fraction between epochs.
            :param exclude_bad: Whether to exclude bad epochs.
            :param baseline: Baseline period for correction.
            :param kwargs: Additional keyword arguments for filtering.
            :return: None if successful, or skips file if an error occurs and miss_bad_data is True.
            """
            try:
                # Retrieve the raw data for processing
                raw_data = get_data_row(row)

                # Apply resampling if specified
                if resample:
                    raw_data.resample(resample, **kwargs)

                # Calculate step size and event intervals
                step_sec = seg_sec * (1 - overlap)
                start = 0 if baseline is None or baseline[0] is None or baseline[0] >= 0 else -baseline[0]

                # Create events for fixed-length segments
                events = mne.make_fixed_length_events(raw_data, start=start, stop=None, duration=step_sec)
                event_id = {'segment': 1}

                # Create epochs from raw data and events
                epochs = mne.Epochs(raw_data, events, event_id, tmin=0, tmax=seg_sec,
                                    baseline=baseline, preload=True, **kwargs)

                # Exclude bad epochs if specified
                if exclude_bad:
                    epochs.drop_bad()

                # Save epochs data as numpy array
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                np.save(os.path.join(output_path, f"{file_name}_pretrain_epoch.npy"), epochs.get_data())

                return None

            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None
                else:
                    raise

        # Batch process the data
        self.batch_process(con_func,
                           lambda row: app_func(row, output_path, seg_sec=seg_sec, resample=resample,
                                                overlap=overlap, exclude_bad=exclude_bad,
                                                baseline=baseline, **kwargs),
                           is_patch=False,
                           result_type=None)

    def get_events(self, miss_bad_data: bool = False, **kwargs) -> None:
        r"""Extract events and log them in the data rows.

        This method processes each data row by applying the `get_data_row()` and `extract_events()`
        functions. The `**kwargs` parameters are passed seamlessly to both functions
        to allow customization of data extraction and event handling.

        Parameters
        ----------
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs. Defaults to `False`.
        **kwargs : dict, optional
            Additional keyword arguments that are passed to both `get_data_row()` and `extract_events()`
            for advanced filtering and event extraction.

        Raises
        ------
        Exception
            If `miss_bad_data` is `False`, an exception is raised on processing errors.

        Note
        ----
        Please refer to the documentation of `get_data_row()` and `extract_events()`
        for detailed descriptions of the available `kwargs` parameters.
        """


        def con_func(row):
            return True

        def app_func(row):
            try:
                # Pass kwargs to get_data_row()
                mne_raw = get_data_row(row, **kwargs)

                # Pass kwargs to extract_events()
                events, event_id = extract_events(mne_raw, **kwargs)

                row["event_id"] = str(event_id)
                event_id_num = {key: sum(events[:, 2] == val) for key, val in event_id.items()}
                row["event_id_num"] = str(event_id_num)
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return None to indicate failure
                else:
                    raise

        # Process all rows and update the locator attribute
        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def epoch_by_event(self, output_path: str, resample: int = None,
                       exclude_bad=True, miss_bad_data=False, **epoch_params):
        r"""Batch process EEG data to create epochs based on events specified in the event_id column.

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        resample : int, optional
            Resample rate for the raw data. If None, no resampling is performed.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Uses simple heuristics to determine bad epochs. Default is `True`.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is `False`.
        **epoch_params : dict, optional
            Additional parameters for `mne.Epochs`, excluding `raw_data`, `events`, and `event_id`.

        Returns
        -------
        None
            The function modifies the dataset in place by creating and saving the epochs.

        Raises
        ------
        ValueError
            If any parameters are inconsistent or if the specified output path is invalid.
        """

        def con_func(row):
            return True

        def app_func(row, output_path: str, resample: int = None,
                     exclude_bad=True, **epoch_params):
            try:
                # Load raw data
                raw_data = get_data_row(row)

                # Resample if needed
                if resample:
                    raw_data.resample(resample)

                # Extract event_id from row
                event_id_str = row.get('event_id')
                # Convert event_id string to dictionary
                events, _ = extract_events(raw_data)
                event_id = ast.literal_eval(event_id_str)
                if not event_id_str or len(event_id) == 0:
                    print(f"No event_id found for file {row['File Path']}")
                    return None

                # Create epochs with the passed epoch_params
                epochs = mne.Epochs(raw_data, events, event_id, **epoch_params)

                # Exclude bad epochs
                if exclude_bad:
                    epochs.drop_bad()

                # Convert epochs to numpy array and save to respective subdirectories
                for event in event_id:
                    event_epochs = epochs[event]
                    file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                    event_output_path = os.path.join(output_path, event)
                    os.makedirs(event_output_path, exist_ok=True)
                    np.save(os.path.join(event_output_path, f"{file_name}_{event}_epoch.npy"), event_epochs.get_data())

                return None
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None
                else:
                    raise

        # Use batch_process to process data
        self.batch_process(con_func,
                           app_func=lambda row: app_func(row, output_path, resample=resample,
                                                         exclude_bad=exclude_bad, **epoch_params),
                           is_patch=False,
                           result_type=None)

    def infer_units(self, miss_bad_data: bool = False, **kwargs: dict) -> None:
        r"""Infer the units of each channel and record them in the data line.

        Parameters
        ----------
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next file when an error occurs.
            Defaults to `False`.

        **kwargs : dict
            Additional parameters passed to `get_data_row()`. These allow for more flexible data processing
            during the inference process. For more details on available options, refer to the `get_data_row()` function.

        Raises
        ------
        Exception
            If an error occurs during file processing and `miss_bad_data` is set to `False`.

        Notes
        -----
        This method applies a custom function to each row in the dataframe to infer the units for each channel
        based on the raw MNE data. The function handles errors gracefully if `miss_bad_data` is `True`.
        """

        def con_func(row):
            return True

        def app_func(row):
            try:
                # Pass **kwargs to get_data_row
                mne_raw = get_data_row(row, **kwargs)

                units = {}

                ch_names = mne_raw.info['ch_names']
                ch_types = mne_raw.get_channel_types()

                for ch_name, ch_type in zip(ch_names, ch_types):
                    ch_data = mne_raw.get_data(picks=[ch_name])

                    # Infer unit for each channel
                    unit = infer_channel_unit(ch_name, ch_data, ch_type)
                    units[ch_name] = unit

                row["Infer Unit"] = str(units)
                return row

            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return None to indicate failure
                else:
                    raise

        new_locator = self.batch_process(
            con_func,
            lambda row: app_func(row),
            is_patch=False,
            result_type='series'
        )

        self.get_shared_attr()['locator'] = new_locator

    def get_quality(self, miss_bad_data: bool = False, **kwargs) -> None:
        r"""Process the data quality of EEG files by calculating quality scores for each row in the dataset.

        Parameters
        ----------
        miss_bad_data : bool, optional
            If `True`, skips rows that contain bad data without raising an error.
            If `False`, raises an exception when encountering bad data.

        **kwargs : dict
            Additional keyword arguments passed to the `get_data_row()` function.
            This allows fine-tuning of parameters such as unit conversion, data normalization, etc.
            For details, refer to the `get_data_row()` function documentation.

        Returns
        -------
        None
            The function modifies the dataset in place by updating quality scores for each row.
        """

        def con_func(row):
            return True

        def app_func(row):
            try:
                # Pass **kwargs to get_data_row to ensure seamless integration
                raw_data = get_data_row(row, **kwargs)
                scores = calculate_eeg_quality_scores(raw_data)
                score = np.mean(scores)
                return str(score)
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return ""  # Return an empty string to indicate failure
                else:
                    raise

        results = self.batch_process(con_func, app_func, is_patch=False, result_type="value")
        self.set_column("Score", results)

    def replace_paths(self, old_prefix, new_prefix):
        r"""Replace the prefix of file paths in the dataset according to the provided mapping.

        Parameters
        ----------
        old_prefix : str
            The old path prefix to be replaced.

        new_prefix : str
            The new path prefix to replace the old one.

        Returns
        -------
        object
            A new UnifiedDataset with updated locator.
        """

        def replace_func(row):
            original_path = row['File Path']
            new_path = original_path
            # Replace the path prefix based on the provided mapping
            if original_path.startswith(old_prefix):
                new_path = original_path.replace(old_prefix, new_prefix, 1)  # Only replace the first occurrence
            row['File Path'] = new_path
            return row

        copied_instance = copy.deepcopy(self)

        # Process the dataset, applying the path replacement function to each row
        updated_locator = self.batch_process(lambda row: True, replace_func, is_patch=False, result_type='series')

        copied_instance.set_shared_attr({'locator': updated_locator})
        return copied_instance
