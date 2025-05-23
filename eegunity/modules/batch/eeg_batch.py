import os
import warnings
import mne
import numpy as np
import pandas as pd
import hashlib
import pickle
from pathlib import Path
from typing import Callable, Union, Tuple, List, Dict, Optional
from eegunity.modules.batch.eeg_scores_shady import compute_quality_scores_shady
from eegunity.modules.batch.eeg_scores_modified_mne import compute_quality_score_mne
from eegunity.modules.parser.eeg_parser import get_data_row, channel_name_parser, extract_events, infer_channel_unit
from eegunity._share_attributes import _UDatasetSharedAttributes
from eegunity.utils.h5 import h5Dataset
from eegunity.utils.handle_errors import handle_errors
from eegunity.utils.log_processing import log_processing
from eegunity.modules.batch.method_mixin_epoch import EEGBatchMixinEpoch


class EEGBatch(_UDatasetSharedAttributes, EEGBatchMixinEpoch):
    """
    This is a key module of `UnifiedDataset` class, with focus on batch processing.
    This `EEGBatch` class has the same attributes as the UnifiedDataset class. In this
    class, we define the functions relative to EEG batch processing.
    """

    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr
        self.main_instance = main_instance

    def batch_process(self,
                      con_func: Callable,
                      app_func: Callable,
                      is_patch: bool,
                      result_type: Union[str, None] = None):
        r"""Process each row of locator based on conditions specified
        in `con_func` and apply `app_func` accordingly. This function handles both list
        and dataframe return types, ensuring the result aligns with the original locator's
        rows based on the `is_patch` flag.

        Parameters
        ----------
        con_func : Callable
            A function that takes a row of locator and returns `True` or `False` to determine
            if `app_func` should be applied to that row. The input is a single row from the
            locator, which you can access like a dictionary. For example, to read the file
            path attribute, use: file_path = row['File Path']
        app_func : Callable
            A function that processes a row of locator and returns the result. The input is same
            as con_func.
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

        Note
        ----
        This method is essential when designing a custom processing pipeline for the dataset.
        Ensure that `con_func` and `app_func` are compatible with the structure of the locator.
        If using `is_patch`, consider the implications on the data integrity.


        Examples
        ---------
        >>> from eegunity import UnifiedDataset
        >>> u_ds = UnifiedDataset(***)
        >>> # example1
        >>> new_locator = u_ds.eeg_batch.batch_process(app_func, con_func, is_patch=True, result_type='series')
        >>> print(new_locator)
        >>> # example2
        >>> a_list = u_ds.eeg_batch.batch_process(app_func, con_func, is_patch=True, result_type='value')
        >>> print(a_list)
        >>> # example3
        >>> u_ds.eeg_batch.batch_process(app_func, con_func, is_patch=True, result_type=None)

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
                warnings.warn(
                    "The file list are empty. Returning an empty Locator. Please check the con_func() and app_func().")
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

    def set_metadata(self, col_name: str, value: list):
        r"""Set the specified metadata in the locator with the given list of values.
        This function is generally used to modify metadata of datasets, directly.

        Parameters
        ----------
        col_name : str
            The name of the column to be set, such as `File Path`, `File Path`, `Domain Tag`, `File Type`, `Data Shape`,
             `Channel Names`, `Number of Channels`, `Sampling Rate`, `Duration`, `Completeness Check`.

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

        Note
        ----
        Ensure that the provided `value` list contains valid entries for the specified column type.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.set_metadata('Sampling Rate', [250, 250, 250])
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
            file_type: Union[str, None] = None,
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

        Note
        ----
        This method modifies the 'locator' dataframe in place based on the provided filters.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.sample_filter(completeness_check='Completed')
        """

        def con_func(row):
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

        # Process the dataframe
        filtered_locator = self.batch_process(con_func, lambda row: row, is_patch=False, result_type='series')

        # Update the locator in shared attributes
        self.set_shared_attr({'locator': filtered_locator})

    def save_as_other(self,
                      output_path: str,
                      domain_tag: Union[str, None] = None,
                      format: str = 'fif',
                      preserve_events: bool = True,
                      get_data_row_params: Dict = None,
                      overwrite: bool = False,
                      miss_bad_data=False) -> 'UnifiedDataset':
        r"""Save data in the specified format ('fif' or 'csv') to the given output path. If you want
        to save as hdf5 file, please use 'export_h5Dataset', because hdf5 file is generally used to
        save the whole dataset.

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
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        overwrite : bool, optional
            If `True`, existing files with the same name will be overwritten. If `False`, a new
            file name with an incremented suffix (e.g., "_raw(1).fif") will be created to avoid
            overwriting. Defaults to `False`.

        Returns
        -------
        None
            This method modifies internal state in-place and does not return any value.
        Raises
        ------
        FileNotFoundError
            If the output path does not exist.
        ValueError
            If the `format` is not 'fif' or 'csv'.

        Note
        ----
        Ensure that the `output_path` is accessible and has the necessary write permissions.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> new_locator = unified_dataset.eeg_batch.save_as_other('/path/to/output', domain_tag='example', format='fif', overwrite=False)
        """
        # Set default empty dictionary if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {'is_set_channel_type': False}
        else:
            get_data_row_params.setdefault('is_set_channel_type', False)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output path does not exist: {output_path}")

        if format not in ['fif', 'csv']:
            raise ValueError(
                f"Unsupported format: {format}. In the current version, only 'fif' and 'csv' are supported.")

        @handle_errors(miss_bad_data)
        def app_func(row):
            raw = get_data_row(row, **get_data_row_params)
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            new_file_path = os.path.join(output_path, f"{file_name}_raw.fif")

            # Check if file exist
            counter = 1
            while os.path.exists(new_file_path) and not overwrite:
                print(f"File '{new_file_path}' already exists.")
                new_file_path = os.path.join(output_path, f"{file_name}({counter})_raw.fif")
                counter += 1

            if format == 'fif':
                raw.save(new_file_path, overwrite=overwrite)
                row['File Path'] = new_file_path
                row['File Type'] = "standard_data"

            elif format == 'csv':
                # Extract data and channel names from Raw object
                data, times = raw.get_data(return_times=True)
                channel_names = raw.info['ch_names']

                # Create a DataFrame with 'timestamp' column and channel data
                df = pd.DataFrame(data.T, columns=channel_names)
                df.insert(0, 'timestamp', times)  # Add 'timestamp' as the first column

                if preserve_events:
                    events, event_id = extract_events(raw)
                    df['marker'] = np.nan
                    for event in events:
                        onset_sample = event[0]
                        event_code = event[2]
                        closest_time_idx = np.argmin(np.abs(times - raw.times[onset_sample]))
                        df.at[closest_time_idx, 'marker'] = event_code
                    row['Number of Channels'] = str(int(row['Number of Channels']) + 1)
                    row['Channel Names'] += ', marker'
                    row['Channel Names'] = 'timestamp, ' + row['Channel Names']
                    data_shape = eval(row['Data Shape'])
                    smaller_shape, larger_shape = min(data_shape), max(data_shape)
                    updated_shape = f"({smaller_shape + 2}, {larger_shape})" if smaller_shape == data_shape[
                        0] else f"({larger_shape}, {smaller_shape + 2})"
                    row['Data Shape'] = updated_shape
                else:
                    row['Channel Names'] = 'timestamp, ' + ', '.join(channel_names)
                    row['Number of Channels'] = str(int(row['Number of Channels']) + 1)
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

        new_locator = self.batch_process(lambda row: domain_tag is None or row['Domain Tag'] == domain_tag,
                                         app_func,
                                         is_patch=False,
                                         result_type='series')
        self.set_shared_attr({'locator': new_locator})
        return None

    def process_mean_std(self,
                         domain_mean: bool = True,
                         pick_type_params: dict = {'eeg': True, 'meg': False, 'stim': False, 'eog': False},
                         miss_bad_data=False) -> None:
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
        pick_type_params : dict, optional
            Additional keyword arguments passed to :func:`mne.pick_types`. This allows users to pass extra parameters
            required by the `mne.pick_types` function seamlessly. For details on the parameters, refer to the
            :func:`mne.pick_types()` function in MNE-Python documentation.
            Detault is `{'eeg':True, 'meg':False, 'stim':False, 'eog':False}`

        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs. Defaults to `False`.


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

        Note
        ----
        Ensure that the EEG data is properly formatted and that all necessary channels are present before calling
        this method.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.process_mean_std(domain_mean=True)
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
            eeg_channel_indices = mne.pick_types(data.info, **pick_type_params)

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

        @handle_errors(miss_bad_data)
        def app_func(row):
            data = get_data_row(row)
            mean_std = get_mean_std(data)
            return mean_std

        results = self.batch_process(lambda row: row['Completeness Check'] != 'Unavailable', app_func, is_patch=False,
                                     result_type="value")

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
                        f"Inconsistent channel names in domain '{domain_tag}', "
                        f"please try domain_mean=False or check the locator manually and make sure all channel names are the same within a domain.")
                num_channels = [len(keys) for keys in all_keys_sets]
                if len(set(num_channels)) != 1:
                    raise ValueError(
                        f"Inconsistent number of channels in domain '{domain_tag}', "
                        f"please try domain_mean=False or check the locator manually and make sure all channel names are the same within a domain.")

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
            self.set_metadata("MEAN STD", new_results)
        else:
            self.set_metadata("MEAN STD", results)

    def format_channel_names(self,
                             format_type: str = 'EEGUnity',
                             miss_bad_data: bool = False):
        r"""Format channel names in the dataset and update the 'Channel Names' column.

        This function processes each row in the dataset, checks the 'Channel Names'
        column, and applies a formatting function to standardize the channel names.
        The function utilizes the `batch_process` method to apply the formatting to
        each row of locator, and the updated channel names are then saved back to
        the 'Channel Names' column.

        Parameters
        ----------
        format_type : str, optional
            The format for channel names, possible values are 'EEGUnity', 'normal', by default 'EEGUnity'. If set to
            'EEGUnity', the channel names are formated in "type:name", like 'EEG:C3', 'EEG:Cz', 'Stim:stim1', which store
            channel type in the locator, rather than change the source data. If set to 'normal', the channel, only formatted
            channels name are stored, like 'C3', 'Cz', 'stim1'.

        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs. Defaults to `False`.
        Returns
        -------
        None
            The function modifies the dataset in place by updating the
            'Channel Names' column.

        Raises
        ------
        KeyError
            If the 'Channel Names' column is missing from the dataset.

        Note
        ----
        Ensure that the dataset is properly loaded and contains the 'Channel Names' column
        before calling this method.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.format_channel_names()
        """

        cache = {}

        @handle_errors(miss_bad_data)
        def app_func(row):
            channel_name = row['Channel Names']
            if channel_name in cache:
                return cache[channel_name]
            else:
                formatted_channel_name = channel_name_parser(channel_name)
                cache[channel_name] = formatted_channel_name
                return formatted_channel_name

        results = self.batch_process(lambda row: True, app_func, is_patch=False, result_type="value")
        self.set_metadata("Channel Names", results)

    def filter(self,
               output_path: str,
               filter_type: str = 'bandpass',
               l_freq: float = None,
               h_freq: float = None,
               notch_freq: float = None,
               auto_adjust_h_freq: bool = True,
               picks: str = 'all',
               miss_bad_data: bool = False,
               get_data_row_params: Dict = None,
               filter_params: Dict = None,
               notch_filter_params: Dict = None) -> None:
        """
        Apply filtering to the data, supporting low-pass, high-pass, band-pass, and notch filters.

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
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        filter_params : dict, optional
            Additional parameters for `mne_raw.filter()`.
        notch_filter_params : dict, optional
            Additional parameters for `mne_raw.notch_filter()`.

        Returns
        -------
        None
            The function modifies the dataset in place.
        """

        @handle_errors(miss_bad_data)
        def app_func(row, l_freq, h_freq, notch_freq, filter_type, output_path, auto_adjust_h_freq, picks,
                     get_data_row_params, filter_params, notch_filter_params):
            # Construct file paths
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            new_file_path = os.path.join(output_path, f"{file_name}_filter.fif")

            # Get the data
            if get_data_row_params is None:
                get_data_row_params = {}
            mne_raw = get_data_row(row, **get_data_row_params)

            # Get sampling frequency and Nyquist frequency
            sfreq = mne_raw.info['sfreq']
            nyquist_freq = sfreq / 2.0
            # Adjust high-pass frequency if it exceeds the Nyquist frequency
            if h_freq is not None and h_freq >= nyquist_freq:
                if auto_adjust_h_freq:
                    warnings.warn(
                        f"High-pass frequency ({h_freq} Hz) is greater than or equal to Nyquist frequency ({nyquist_freq} Hz). Adjusting h_freq to {nyquist_freq - 1} Hz.")
                    h_freq = nyquist_freq - 1
                else:
                    raise ValueError(
                        f"High-pass frequency must be less than Nyquist frequency ({nyquist_freq} Hz).")

            if notch_freq is not None and notch_freq >= nyquist_freq:
                if auto_adjust_h_freq:
                    warnings.warn(
                        f"Notch frequency ({notch_freq} Hz) is greater than or equal to Nyquist frequency ({nyquist_freq} Hz). Adjusting h_freq to {nyquist_freq - 1} Hz.")
                    notch_freq = nyquist_freq - 1
                else:
                    raise ValueError(
                        f"Notch frequency must be less than Nyquist frequency ({nyquist_freq} Hz).")

            # Apply the appropriate filter based on filter_type
            if filter_type in ['lowpass', 'highpass', 'bandpass']:
                # Prepare filter parameters
                if filter_params is None:
                    filter_params = {}
                filter_kwargs = filter_params.copy()
                filter_kwargs['l_freq'] = l_freq
                filter_kwargs['h_freq'] = h_freq
                filter_kwargs['picks'] = picks
                filter_kwargs.setdefault('fir_design', 'firwin')  # Use 'firwin' if not specified

                mne_raw.filter(**filter_kwargs)
            elif filter_type == 'notch':
                if notch_freq is not None:
                    if notch_filter_params is None:
                        notch_filter_params = {}
                    notch_kwargs = notch_filter_params.copy()
                    notch_kwargs['freqs'] = notch_freq
                    notch_kwargs['picks'] = picks
                    mne_raw.notch_filter(**notch_kwargs)
                else:
                    raise ValueError("notch_freq must be specified for notch filter.")
            else:
                raise ValueError("Invalid filter_type. Must be 'lowpass', 'highpass', 'bandpass', or 'notch'.")

            # Save filtered data
            mne_raw.save(new_file_path, overwrite=True)
            row['File Path'] = new_file_path

            return None

        # Process the batch
        self.batch_process(lambda row: True,
                           lambda row: app_func(row, l_freq, h_freq, notch_freq, filter_type,
                                                output_path, auto_adjust_h_freq, picks,
                                                get_data_row_params, filter_params, notch_filter_params),
                           is_patch=False,
                           result_type=None)

        # Update file paths in the dataset
        self.get_shared_attr()["dataset_path"] = output_path
        self.set_shared_attr({'locator': self.main_instance.eeg_parser.check_locator(
            self.main_instance.eeg_parser._process_directory(output_path))})

    def ica(self, output_path: str, miss_bad_data: bool = False, get_params: Dict = None,
            ica_params: Dict = None, fit_params: Dict = None, apply_params: Dict = None):
        r"""Apply ICA (Independent Component Analysis) to the specified file in the dataset.

        This method applies ICA to clean the EEG data using parameters passed through `ica_params`, `fit_params`,
        and `apply_params`. Please refer to the official documentation for `mne.preprocessing.ICA`, `ica.fit()`,
        and `ica.apply()` for the complete list of available parameters.

        Documentation links:
        - `mne.preprocessing.ICA`: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
        - `ICA.fit`: https://mne.tools/stable/generated/mne.preprocessing.ICA.html#fitting-ica
        - `ICA.apply`: https://mne.tools/stable/generated/mne.preprocessing.ICA.html#applying-ica

        Parameters
        ----------
        output_path : str
            Path to save the processed file after applying ICA.
        miss_bad_data : bool, optional
            Whether to skip bad data files and continue processing the next one. Defaults to `False`.
        get_params : dict, optional
            Additional parameters passed to `eegunity.module_eeg_parser.eeg_parser.get_data_row`,
        ica_params : dict, optional
            Additional parameters passed to `mne.preprocessing.ICA`, such as `n_components`, `method`, etc.
        fit_params : dict, optional
            Additional parameters passed to `ica.fit()`, such as `picks`, `decim`, etc.
        apply_params : dict, optional
            Additional parameters passed to `ica.apply()`, such as `exclude`, `include`, etc.

        Returns
        -------
        None
            Updates the file path in the dataset locator after ICA is applied.

        Raises
        ------
        ValueError
            If the output path is invalid or if the specified parameters are inconsistent.

        Note
        ----
        Ensure that the input data is properly formatted and that all necessary parameters are specified
        before calling this method.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.ica('/path/to/save/', ica_params={'n_components': 20}, fit_params={'picks': 'eeg'})
        """

        @handle_errors(miss_bad_data)
        def app_func(row, output_path: str):
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            new_file_path = os.path.join(output_path, f"{file_name}_ica_cleaned_raw.fif")
            mne_raw = get_data_row(row, **(get_params or {}))

            # Initialize ICA with specific parameters
            ica = mne.preprocessing.ICA(**(ica_params or {}))

            # Fit ICA with specific parameters
            ica.fit(mne_raw, **(fit_params or {}))

            # Apply ICA to raw data with specific parameters
            ica.apply(mne_raw, **(apply_params or {}))

            # Save file and update path in locator
            mne_raw.save(new_file_path, overwrite=True)
            row['File Path'] = new_file_path

            return new_file_path

        new_path_list = self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(row, output_path),
            is_patch=False,
            result_type='value'
        )

        # Update locator
        self.set_metadata("File Path", new_path_list)
        locator_df = self.get_shared_attr()['locator']
        locator_df = locator_df[locator_df['File Path'] != ""]
        self.get_shared_attr()['locator'] = locator_df

    def resample(self, output_path: str, miss_bad_data: bool = False, resample_params: Dict = None,
                 get_data_row_params: Dict = None) -> None:
        r"""Resample the data using MNE's resampling functionality and save the processed data.

        Parameters
        ----------
        output_path : str
            The path where the resampled file will be saved.
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs. Defaults to `False`.
        resample_params : dict, optional
            Additional parameters to be passed to the `mne_raw.resample()` function.
        get_data_row_params : dict, optional
            Additional parameters to be passed to the `get_data_row()` function.

        Returns
        -------
        None
            The function modifies the dataset in place by saving the resampled data.

        Raises
        ------
        Exception
            If an error occurs during resampling and `miss_bad_data` is set to `False`, the error will be raised.

        Note
        ----
        Ensure that the output path is accessible and that the input data is properly formatted before calling this method.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.resample('/path/to/save/', resample_params={'sfreq': 256})
        """

        # Set default empty dictionaries if parameters are None
        if resample_params is None:
            resample_params = {}
        if get_data_row_params is None:
            get_data_row_params = {}

        @handle_errors(miss_bad_data)
        def app_func(row, output_path: str) -> dict:
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            new_file_path = os.path.join(output_path, f"{file_name}_resampled_raw.fif")

            mne_raw = get_data_row(row, **get_data_row_params)

            # Resample using the provided resample_params
            mne_raw.resample(**resample_params)

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

        self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(row, output_path),
            is_patch=False,
            result_type=None
        )
        # Update file paths in the dataset
        self.get_shared_attr()["dataset_path"] = output_path
        self.set_shared_attr({'locator': self.main_instance.eeg_parser.check_locator(
            self.main_instance.eeg_parser._process_directory(output_path))})

    def align_channel(self, output_path: str, channel_order: list, min_num_channels: int = 1,
                      miss_bad_data: bool = False, get_data_row_params: Dict = None) -> None:
        r"""Adjust the channel order and perform interpolation on the data.

        This method realigns the EEG data channels based on the provided `channel_order`. It utilizes
        `get_data_row()` for retrieving the data. Additional parameters can be passed to `get_data_row()`
        via `get_data_row_params`. For more information on available options, refer to the
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
        get_data_row_params : dict, optional
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

        Note
        ----
        Ensure that the output path is accessible and that the provided channel order is valid before calling this method.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.align_channel('/path/to/save/', channel_order=['C3', 'C4', 'O1'], min_num_channels=3)
        """
        from eegunity.utils import channel_align_raw
        if get_data_row_params is None:
            get_data_row_params = {}
        @handle_errors(miss_bad_data)
        def app_func(row, channel_order, output_path, min_num_channels):
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            new_file_path = os.path.join(output_path, f"{file_name}_aligned.fif")

            # Fetch the data using get_data_row and get_data_row_params
            mne_raw = get_data_row(row, **get_data_row_params)

            aligned_raw = channel_align_raw(mne_raw, channel_order, min_matched_channel=min_num_channels)

            aligned_raw.save(new_file_path, overwrite=True)
            row['File Path'] = new_file_path
            row['Channel Names'] = ', '.join(aligned_raw.info['ch_names'])
            row['Number of Channels'] = str(len(channel_order))
            dimensions = row['Data Shape'].strip('()').split(',')
            dimensions = [int(dim.strip()) for dim in dimensions]
            dimensions[dimensions.index(min(dimensions))] = len(channel_order)
            row['Data Shape'] = f"({dimensions[0]}, {dimensions[1]})"
            return row

        new_locator = self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(row, channel_order, output_path, min_num_channels),
            is_patch=False,
            result_type='series'
        )
        self.get_shared_attr()['locator'] = new_locator

    def normalize(self, output_path: str, norm_type: str = 'sample-wise', miss_bad_data: bool = False,
                  domain_mean: bool = True, get_data_row_params: Dict = None) -> None:
        r"""Normalize the data.

        This method normalizes the EEG data based on the specified normalization type. It can either
        perform sample-wise normalization or aggregate by domain mean, depending on the provided parameters.

        Parameters
        ----------
        output_path : str
            The path where the normalized file will be saved.
        norm_type : str
            The type of normalization to perform. It can be:
            - 'channel-wise': Normalize each channel individually based on its mean and standard deviation.
            - 'sample-wise': Normalize all channels based on a common mean and standard deviation.
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs.
            Defaults to `False`.
        domain_mean : bool, optional
            If True (default), the function aggregates the results by domain tags. Each domain contains the
            mean and standard deviation across all related EEG channels. If False, the function calculates
            and stores individual mean and standard deviation for each EEG recording.
        get_data_row_params : dict, optional
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

        Note
        ----
        Ensure that the output path is accessible and that the input data is properly formatted before calling this method.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.normalize('/path/to/save/', norm_type='channel-wise', domain_mean=True)
        """

        # Set default empty dictionary if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}

        # Check if 'MEAN STD' column exists, if not process mean and std
        locator = self.get_shared_attr()['locator']
        if 'MEAN STD' not in locator.columns:
            self.process_mean_std(domain_mean=domain_mean)

        @handle_errors(miss_bad_data)
        def app_func(row, norm_type, output_path):
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            new_file_path = os.path.join(output_path, f"{file_name}_normed.fif")

            # Include norm_type in get_data_row_params
            get_data_row_params_with_norm = get_data_row_params.copy()
            get_data_row_params_with_norm['norm_type'] = norm_type

            mne_raw = get_data_row(row, **get_data_row_params_with_norm)
            mne_raw.save(new_file_path, overwrite=True)
            row['File Path'] = new_file_path
            row['File Type'] = "standard_data"
            return row

        new_locator = self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(row, norm_type, output_path),
            is_patch=False,
            result_type='series'
        )
        self.get_shared_attr()['locator'] = new_locator

    def epoch_for_pretraining(self,
                              output_path: str,
                              seg_sec: float,
                              resample: Optional[int] = None,
                              overlap: float = 0.0,
                              exclude_bad: bool = True,
                              baseline: Tuple[Optional[float], float] = None,
                              miss_bad_data: bool = False,
                              get_data_row_params: Dict = None,
                              resample_params: Dict = None,
                              epoch_params: Dict = None) -> None:
        r"""Processes raw EEG data by creating epochs for pretraining, with optional resampling and event segmentation.

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
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        resample_params : dict, optional
            Additional parameters passed to `raw_data.resample()`.
        epoch_params : dict, optional
            Additional parameters passed to `mne.Epochs()`.

        Returns
        -------
        None
            The function modifies the dataset in place by saving the processed epoch data.

        Raises
        ------
        ValueError
            If the segment length is invalid or if any specified parameters are inconsistent.

        Note
        ----
        Ensure that the output path is accessible and that the input data is properly formatted before calling this method.

        Examples
        --------
        >>> from eegunity import UnifiedDataset
        >>> unified_dataset = UnifiedDataset(***)
        >>> unified_dataset.eeg_batch.epoch_for_pretraining('/path/to/save/', seg_sec=2.0, resample=256)
        """

        # Set default empty dictionaries if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}
        if resample_params is None:
            resample_params = {}
        if epoch_params is None:
            epoch_params = {}

        @handle_errors(miss_bad_data)
        def app_func(row: Dict,
                     output_path: str,
                     seg_sec: float,
                     resample: Optional[int] = None,
                     overlap: float = 0.0,
                     exclude_bad: bool = True,
                     baseline: Tuple[Optional[float], float] = None,
                     get_data_row_params: Dict = None,
                     resample_params: Dict = None,
                     epoch_params: Dict = None) -> Optional[None]:
            """
            Applies the epoch processing to a single row of data.

            Parameters
            ----------
            row : dict
                A dictionary representing a data row, including 'File Path' for raw EEG file.
            output_path : str
                Path to save the epoch data.
            seg_sec : float
                Length of each epoch in seconds.
            resample : int, optional
                Optional resampling rate.
            overlap : float, optional
                Overlap fraction between epochs.
            exclude_bad : bool, optional
                Whether to exclude bad epochs.
            baseline : tuple, optional
                Baseline period for correction.
            get_data_row_params : dict, optional
                Additional parameters for `get_data_row()`.
            resample_params : dict, optional
                Additional parameters for `raw_data.resample()`.
            epoch_params : dict, optional
                Additional parameters for `mne.Epochs()`.
            """
            # Retrieve the raw data for processing
            raw_data = get_data_row(row, **get_data_row_params)

            # Apply resampling if specified
            if resample:
                raw_data.resample(resample, **resample_params)

            # Calculate step size and event intervals
            step_sec = seg_sec * (1 - overlap)
            start = 0 if baseline is None or baseline[0] is None or baseline[0] >= 0 else -baseline[0]

            # Create events for fixed-length segments
            events = mne.make_fixed_length_events(raw_data, start=start, stop=None, duration=step_sec)
            event_id = {'segment': 1}
            # Create epochs from raw data and events
            epochs = mne.Epochs(raw_data, events, event_id, tmin=0, tmax=seg_sec,
                                baseline=baseline, preload=True, **epoch_params)

            # Exclude bad epochs if specified
            if exclude_bad:
                epochs.drop_bad()

            # Save epochs data as numpy array
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            np.save(os.path.join(output_path, f"{file_name}_pretrain_epoch.npy"), epochs.get_data())

            return None

        # Batch process the data
        self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(
                row,
                output_path,
                seg_sec=seg_sec,
                resample=resample,
                overlap=overlap,
                exclude_bad=exclude_bad,
                baseline=baseline,
                get_data_row_params=get_data_row_params,
                resample_params=resample_params,
                epoch_params=epoch_params
            ),
            is_patch=False,
            result_type=None
        )

    def get_events(self, miss_bad_data: bool = False, get_data_row_params: Dict = None) -> None:
        r"""Extract events and log them in the data rows.

        This method processes each data row by applying the `get_data_row()` and `extract_events()`
        functions. Additional parameters can be passed to these functions via `get_data_row_params` and `extract_events_params`.

        Parameters
        ----------
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next one if an error occurs. Defaults to `False`.
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.

        Raises
        ------
        Exception
            If `miss_bad_data` is `False`, an exception is raised on processing errors.

        Note
        ----
        Please refer to the documentation of `get_data_row()` and `extract_events()`
        for detailed descriptions of the available parameters.
        """

        # Set default empty dictionaries if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}

        @handle_errors(miss_bad_data)
        def app_func(row):
            # Pass get_data_row_params to get_data_row()
            mne_raw = get_data_row(row, preload=False, **get_data_row_params)

            # Pass extract_events_params to extract_events()
            events, event_id = extract_events(mne_raw)

            row["event_id"] = str(event_id)
            event_id_num = {key: sum(events[:, 2] == val) for key, val in event_id.items()}
            row["event_id_num"] = str(event_id_num)
            return row

        # Process all rows and update the locator attribute
        new_locator = self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(row),
            is_patch=False,
            result_type='series'
        )
        self.get_shared_attr()['locator'] = new_locator

    def infer_units(self, miss_bad_data: bool = False, get_data_row_params: Dict = None) -> None:
        r"""Infer the units of each channel and record them in the data line.

        Parameters
        ----------
        miss_bad_data : bool, optional
            Whether to skip the current file and continue processing the next file when an error occurs.
            Defaults to `False`.
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()`. These allow for more flexible data processing
            during the inference process.

        Raises
        ------
        Exception
            If an error occurs during file processing and `miss_bad_data` is set to `False`.

        Note
        ----
        This method applies a custom function to each row in the dataframe to infer the units for each channel
        based on the raw MNE data. The function handles errors gracefully if `miss_bad_data` is `True`.
        """

        # Set default empty dictionary if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}

        @handle_errors(miss_bad_data)
        def app_func(row):
            # Pass get_data_row_params to get_data_row
            mne_raw = get_data_row(row, **get_data_row_params)

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

        new_locator = self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(row),
            is_patch=False,
            result_type='series'
        )

        self.get_shared_attr()['locator'] = new_locator

    def get_quality(self, miss_bad_data: bool = False,
                    method: str = 'shady',
                    ica_params: Dict = None,
                    save_name: str = 'scores',
                    get_data_row_params: Dict = None) -> None:
        r"""Process the data quality of EEG files by calculating quality scores for each row in the dataset.

        Parameters
        ----------
        miss_bad_data : bool, optional
            If `True`, skips rows that contain bad data without raising an error.
            If `False`, raises an exception when encountering bad data.

        get_data_row_params : dict, optional
            Additional parameters passed to the `get_data_row()` function.
            This allows fine-tuning of parameters such as unit conversion, data normalization, etc.
            For details, refer to the `get_data_row()` function documentation.

        Returns
        -------
        None
            The function modifies the dataset in place by updating quality scores for each row.
        """

        # Set default empty dictionary if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}

        @handle_errors(miss_bad_data)
        def app_func(row):
            # Pass get_data_row_params to get_data_row to ensure seamless integration
            raw_data = get_data_row(row, **get_data_row_params)
            if method == 'shady':
                scores = compute_quality_scores_shady(raw_data)
                score = np.mean(scores)
            elif method == 'ica':
                score = compute_quality_score_mne(raw_data, ica_params=ica_params)
            else:
                raise ValueError('Paramether method must be "shady", "ica"')
            return str(score)

        results = self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            app_func, is_patch=True, result_type="value")
        self.set_metadata(save_name, results)

    def replace_paths(self, old_prefix, new_prefix):
        r"""Replace the prefix of file paths in the dataset according to the provided mapping (in-place).
        "This function is generally used in the context of multi-server or multi-user coordination.

        Parameters
        ----------
        old_prefix : str
            The old path prefix to be replaced.

        new_prefix : str
            The new path prefix to replace the old one.

        Returns
        -------
        None
            This method modifies internal state in-place and does not return any value.
        """

        def replace_func(row):
            original_path = row['File Path']
            new_path = original_path
            # Replace the path prefix based on the provided mapping
            if original_path.startswith(old_prefix):
                new_path = original_path.replace(old_prefix, new_prefix, 1)  # Only replace the first occurrence
            row['File Path'] = new_path
            return row

        # Process the dataset, applying the path replacement function to each row
        updated_locator = self.batch_process(lambda row: True, replace_func, is_patch=False, result_type='series')

        self.set_shared_attr({'locator': updated_locator})
        return None

    def export_h5Dataset(self, output_path: str, name: str = 'EEGUnity_export',
                         get_data_row_params: Dict = None, miss_bad_data: bool = False) -> None:
        """
        Export the dataset in HDF5 format to the specified output path, use in this large brain model project:
        https://github.com/935963004/LaBraM;
        Reference: W.-B. Jiang, L.-M. Zhao, and B.-L. Lu, "Large brain model for learning generic representations with
        tremendous EEG data in BCI," Proc. The 12th Int. Conf. Learning Representations, 2024. [Online]. Available:
        https://openreview.net/forum?id=QzTpTRVtrP.

        This function processes all files in the dataset, ensuring that each file
        is stored in a separate group with its own dataset and attributes.

        The exported HDF5 file will have the following structure:

        - A root group named after the provided `name` (default: 'EEGUnity_export').
        - Each file in the dataset is stored as a separate group within the root group.
          - The group name is derived from the basename of the file path (e.g., 'file1.fif').
          - Within each group:
            - A dataset named 'eeg' contains the EEG data (stored as a NumPy array).
            - A dataset named 'info' contains additional metadata about the EEG data, serialized as a uint8 array (using pickle).
            - Attributes for each dataset:
              - 'rsFreq': The sampling rate of the EEG data for the specific file.
              - 'chOrder': The list of channel names in the order they appear in the EEG data.

        Parameters
        ----------
        output_path : str
            The directory path where the exported HDF5 files will be saved.
            A `FileNotFoundError` is raised if the path does not exist.
        name : str
            The name of the HDF5 file. Must be a string. The default value is 'EEGUnity_export'.
            Raises a `TypeError` if the value provided is not a string.
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        miss_bad_data : bool, optional
            If `True`, skips rows that contain bad data without raising an error.
            If `False`, raises an exception when encountering bad data.
        Returns
        -------
        None
            The function does not return any value.

        Raises
        ------
        FileNotFoundError
            If the output path does not exist.
        FileExistsError
            If the HDF5 file already exists in the specified output path.
        ValueError
            If channel configurations or counts are inconsistent within a domain tag.
        TypeError
            If the `name` parameter is not a string.
        """
        # Set default empty dictionary if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}

        if not isinstance(name, str):
            raise TypeError(f"The name parameter must be a string, got {type(name).__name__} instead.")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output path does not exist: {output_path}")

        # Create the HDF5 file path
        h5_path = Path(output_path) / f"{name}.hdf5"

        # Check if the HDF5 file already exists
        if h5_path.exists():
            raise FileExistsError(
                f"The HDF5 file '{h5_path}' already exists. Please choose a different name or delete the existing file.")

        # Create the HDF5 file handler
        dataset = h5Dataset(Path(output_path), name)

        @handle_errors(miss_bad_data=miss_bad_data)
        @log_processing
        def app_func(row):
            # Extract channel information for the current file
            current_channels = [channel.strip() for channel in row['Channel Names'].split(',')]
            current_sf = int(float(row['Sampling Rate']))

            # Get EEG data with additional parameters
            raw = get_data_row(row, **get_data_row_params)
            eeg_data = raw.get_data()

            # Create a group and dataset in the HDF5 file for each file
            grp = dataset.addGroup(grpName=os.path.basename(row['File Path']))
            dset = dataset.addDataset(grp, 'eeg', eeg_data, chunks=eeg_data.shape)

            # Add individual attributes for each dataset
            dataset.addAttributes(dset, 'rsFreq', current_sf)  # Sampling rate may vary per file
            dataset.addAttributes(dset, 'chOrder', current_channels)  # Channel order may vary per file

            # Handle info attribute using pickle

            info_bytes = pickle.dumps(raw.info)
            # Convert bytes to a NumPy array of uint8
            info_array = np.frombuffer(info_bytes, dtype='uint8')
            # Store the serialized info as a uint8 array
            dataset.addDataset(grp, 'info', info_array, chunks=None)

            return None  # No need to return any result

        # Process batch data without domain_tag filtering, processing all files
        self.batch_process(lambda row: row['Completeness Check'] != 'Unavailable',
                           app_func, is_patch=False, result_type=None)

        # Save the dataset to disk
        dataset.save()
        # Print completion message if verbose is True
        print(f"All data exported successfully to {h5_path}.")

    def auto_domain(self) -> None:
        """Automatically modify the 'Domain Tag' of each row based on 'Sampling Rate' and channel names.

        This function processes each row in the dataset and updates the 'Domain Tag'
        by appending the 'Sampling Rate' and a unique encoded representation of the channel names.
        The channel names are retrieved using `get_data_row()` to ensure accuracy,
        and parameters can be passed to `get_data_row()` via `get_data_row_params`.

        The 'Domain Tag' is updated in the format:
        `f"row['Domain Tag']-row['Sampling Rate']Hz-ch_enc(channel_names)"`.

        The function utilizes the `batch_process` method to apply these modifications
        across the dataset.


        Returns
        -------
        None
            The function modifies the dataset in place by updating the 'Domain Tag' column.

        Raises
        ------
        KeyError
            If the required columns ('Domain Tag', 'Sampling Rate') are missing.

        Examples
        --------
        >>> unified_dataset.eeg_batch.auto_domain(get_data_row_params={'preload': True})
        """

        def ch_enc(channel_names: List[str]) -> str:
            """Encodes the channel names into a short unique identifier.

            The identifier includes the length of the channel names list and
            a 4-character hash representing the content. Channel name order does not
            affect the hash, ensuring only differences in content cause a different hash.
            """
            # Sort the list
            channels = sorted(channel_names)
            length = len(channels)

            # Join the sorted channel names back into a single string
            sorted_channel_str = ','.join(channels)

            # Hash the sorted channel string and take the first 4 characters of the hash
            hash_object = hashlib.sha1(sorted_channel_str.encode('utf-8'))
            hash_part = hash_object.hexdigest()[:4]  # Use the first 4 characters of the hash

            # Combine the length of the list with the 4-character hash
            return f"{length}-{hash_part}"

        def app_func(row):
            # Get the necessary values from the row
            domain_tag = row['Domain Tag']
            sampling_rate = row['Sampling Rate']
            channel_names = [ch.strip() for ch in row['Channel Names'].split(',')]
            # Generate the new domain tag using the provided format
            new_domain_tag = f"{domain_tag}-{sampling_rate}Hz-{ch_enc(channel_names)}"
            return new_domain_tag

        # Use batch_process to apply the function to each row
        results = self.batch_process(lambda row: row['Completeness Check'] != 'Unavailable', app_func, is_patch=True,
                                     result_type="value")

        # Update the 'Domain Tag' column with the new values
        self.set_metadata("Domain Tag", results)

    def get_file_hashes(self) -> None:
        """
        Generate and store unique file identifiers for EEG data files.

        This method processes each row in the dataset by reading the file at the
        specified path and computing its SHA-256 hash. The hash serves as a unique
        identifier for the file, which is then stored in the metadata under the
        key "Source Hash".

        The method uses `batch_process` to apply the hash function to all rows,
        and updates the metadata using `set_metadata`.

        Raises:
            FileNotFoundError: If a file at the specified path cannot be found.
            IOError: If a file cannot be read due to permission or corruption issues.
        """

        def app_func(row):
            """
            Compute the SHA-256 hash of a file's content.

            Args:
                row (dict): A dictionary containing file metadata, must include
                            the key 'File path' with the path to the file.

            Returns:
                str: The SHA-256 hash of the file content as a hexadecimal string.
            """
            file_path = row['File Path']

            sha256_hash = hashlib.sha256()
            try:
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
            except IOError as e:
                raise IOError(f"Cannot read file {file_path}: {e}")

            return sha256_hash.hexdigest()

        # Apply the app_func to each row and collect the resulting hash values
        results = self.batch_process(
            lambda row: True,  # Process all rows
            app_func,
            is_patch=True,
            result_type="value"
        )

        # Store the results in the metadata under the key "Source Hash"
        self.set_metadata("Source Hash", results)