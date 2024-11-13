import ast
import datetime
import glob
import json
import os
import re
import warnings
import zipfile
import mne
import numpy as np
import pandas as pd
import scipy
from typing import Union, Dict
from collections import OrderedDict

from eegunity.module_eeg_parser.eeg_parser_csv import process_csv_files
from eegunity.module_eeg_parser.eeg_parser_mat import process_mat_files, _find_variables_by_condition, \
    _condition_source_data
from eegunity.share_attributes import UDatasetSharedAttributes

current_dir = os.path.dirname(__file__)
json_file_path = os.path.join(current_dir, 'combined_montage.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)
STANDARD_EEG_CHANNELS = sorted(data.keys(), key=len, reverse=True)
EEG_PREFIXES_SUFFIXES = ["EEG", "REF", "LE"]


class EEGParser(UDatasetSharedAttributes):
    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr
        dataset_path = self.get_shared_attr()['dataset_path']
        locator_path = self.get_shared_attr()['locator_path']
        if dataset_path and locator_path:
            raise ValueError("The 'datasets' and 'locator' paths cannot both be provided simultaneously.")
        elif not dataset_path and not locator_path:
            raise ValueError("One of 'datasets' or 'locator' paths must be provided.")

        if self.get_shared_attr()['locator_path']:  # initiate UnifiedDataset via Locator
            if os.path.isfile(locator_path) and locator_path.endswith('.csv'):
                self.locator_path = locator_path
                self.set_shared_attr({'locator': self.check_locator(pd.read_csv(locator_path))})
            else:
                raise ValueError(f"The provided 'locator' path {locator_path} is not a valid CSV file.")
        elif self.get_shared_attr()['dataset_path']:  # Construct UnifiedDataset by reading dataset path
            if os.path.isdir(dataset_path):
                self._unzip_if_no_conflict(dataset_path)
                self.set_shared_attr({'locator': self.check_locator(self._process_directory(dataset_path))})
            else:
                raise ValueError("The provided 'datasets' path is not a valid directory.")

    def _process_directory(self, datasets_path, use_relative_path=False):
        """
        Process a directory to gather information on various data files.

        Parameters
        ----------
        datasets_path : str
            The path to the directory containing the dataset files.
        use_relative_path : bool, optional
            Whether to use relative paths instead of absolute paths. Default is False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing metadata for each file in the directory, including file path, domain tag, file type, data shape, channel names, number of channels, sampling rate, duration, and completeness check.
        """
        files_info = []
        datasets_path = os.path.abspath(datasets_path) if not use_relative_path else os.path.relpath(datasets_path)

        for filepath in glob.glob(datasets_path + '/**/*', recursive=True):
            if os.path.isfile(filepath):
                files_info.append([filepath, self.get_shared_attr()['domain_tag'], '', '', '', '', '', '', ''])
        files_locator = pd.DataFrame(files_info,
                                     columns=['File Path', 'Domain Tag', 'File Type', 'Data Shape', 'Channel Names',
                                              'Number of Channels',
                                              'Sampling Rate', 'Duration', 'Completeness Check'])
        files_locator = process_mne_files(files_locator, self.get_shared_attr()['verbose'])
        files_locator = process_mat_files(files_locator)
        files_locator = process_csv_files(files_locator)
        files_locator = _clean_sampling_rate_(files_locator)
        files_locator = files_locator.sort_values(by='File Path').reset_index(drop=True)
        return files_locator

    def _unzip_if_no_conflict(self, datasets_path):
        """
        Unzip zip files in the specified directory if no conflict exists.

        Parameters
        ----------
        datasets_path : str
            The path to the directory where zip files will be searched and extracted.

        Returns
        -------
        None
            The function does not return any value. It performs extraction as a side effect.
        """
        if self.get_shared_attr()['is_unzip']:
            # Recursively traverse all files and subdirectories in the directory
            for root, dirs, files in os.walk(datasets_path):
                for filename in files:
                    # Check if the file is a zip file
                    if filename.endswith('.zip'):
                        file_path = os.path.join(root, filename)
                        # Check if there is a file with the same name after unzipping the zip file
                        # Typically, this checks whether a file exists with the same name as the zip file but without the .zip extension
                        if not os.path.exists(os.path.splitext(file_path)[0]):
                            # No file with the same name exists, so unzip the zip file
                            try:
                                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                    zip_ref.extractall(root)
                                    print(f"Extracted: {file_path}")
                            except zipfile.BadZipFile:
                                print(f"Bad zip file: {file_path}")
                            except Exception as e:
                                print(f"Failed to extract {file_path}: {e}")
                        else:
                            print(f"Skipped {file_path}, conflict exists.")
        else:
            return

    def get_data(self, data_idx, **kwargs):
        """
        Retrieve data based on the specified index from the locator.

        Parameters
        ----------
        data_idx : int
            Index of the row in the locator DataFrame to retrieve data from.


        Returns
        -------
        Any
            The data retrieved and processed according to the specified parameters.
        """
        row = self.get_shared_attr()['locator'].iloc[data_idx]
        return get_data_row(row, **kwargs)

    def check_locator(self, locator):
        """
        Validate the contents of the locator DataFrame.

        Parameters
        ----------
        locator : pd.DataFrame
            A DataFrame containing file metadata, including data shape, channel names, file type, file path, number of channels, sampling rate, and duration.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with a 'Completeness Check' column indicating whether the validation was completed or if errors were found.
        """
        locator = locator.astype(str)

        def check_data_shape(data_shape):
            if data_shape.strip() == '':
                return ["Miss data in Data Shape"]
            try:
                dimensions = [int(dim) for dim in data_shape.strip('()').split(',')]
                if len(dimensions) != 2:
                    return ["Raw Data Shape is not two-dimensional"]
            except ValueError:
                return ["Data Shape format error"]
            return []

        def check_channel_duplicates(channel_names):
            if channel_names.strip() == '':
                return ["Miss data in Channel Names"]
            if channel_names.strip() == 'nan':
                return ["channel name is nan"]
            if pd.isna(channel_names):
                return ["channel name is nan"]
            if pd.isnull(channel_names):
                return ["channel name is null"]

            channel_names = channel_names.split(',')
            seen = set()
            duplicates = set()
            for name in channel_names:
                if name in seen:
                    duplicates.add(name)
                else:
                    seen.add(name)
            if duplicates:
                return [f"Duplicate channels: {', '.join(duplicates)}"]
            return []

        def check_channel_counts(data_shape, num_channels):
            try:
                dimensions = [int(dim) for dim in data_shape.strip('()').split(',')]
                if min(dimensions) != int(float(num_channels)):
                    return ["Mismatch in reported and actual channel counts"]
            except ValueError:
                return ["Channel count or data shape format error"]
            return []

        def check_duration(sampling_rate, duration, data_shape):
            try:
                dimensions = [int(dim) for dim in data_shape.strip('()').split(',')]
                calculated_duration = max(dimensions) / float(sampling_rate)
                if abs(calculated_duration - float(duration)) >= 1:
                    return ["Incorrect duration calculation"]
            except ValueError:
                return ["Duration or sampling rate format error"]
            return []

        for index, row in locator.iterrows():
            errors = []
            errors.extend(check_data_shape(row.get('Data Shape', '').strip()))

            if row.get('File Type', '').strip() == '':
                errors.append("Miss data in File Type")
            elif row['File Type'] == "unidentified":
                errors.append("File Type is 'unidentified'")

            if row.get('File Path', '').strip() == '':
                errors.append("Miss data in File Path")
            elif not os.path.exists(row['File Path']):
                errors.append("File does not exist")

            errors.extend(
                check_channel_counts(row.get('Data Shape', '').strip(), row.get('Number of Channels', '').strip()))
            errors.extend(check_duration(row.get('Sampling Rate', '').strip(), row.get('Duration', '').strip(),
                                         row.get('Data Shape', '').strip()))
            errors.extend(check_channel_duplicates(row.get('Channel Names', '').strip()))
            # update check column
            locator.at[index, 'Completeness Check'] = "Completed" if not errors else "Unavailable"
        return locator


# static function defining
def normalize_data(raw_data, mean_std_str: Union[str, Dict], norm_type: str):
    """
    Normalize EEG data based on provided mean and standard deviation values.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw EEG data to be normalized. The data should be in MNE Raw format.
    mean_std_str : Union[str, Dict]
        A dictionary or string that contains mean and standard deviation values.
        If it's a string, it will be evaluated into a dictionary.
        The dictionary keys should be channel names (for channel-wise normalization)
        or 'all_eeg' (for sample-wise normalization).
    norm_type : str
        The type of normalization to perform. It can be:
        - 'channel-wise': Normalize each channel individually based on its mean and standard deviation.
        - 'sample-wise': Normalize all channels based on a common mean and standard deviation.

    Returns
    -------
    mne.io.Raw
        The normalized raw EEG data.

    Raises
    ------
    ValueError
        If `norm_type` is not 'channel-wise' or 'sample-wise'.
    """

    # If mean_std_str is a string, process it accordingly
    if isinstance(mean_std_str, str):
        mean_std_str = mean_std_str.replace('nan', 'None')
        mean_std_dict = ast.literal_eval(mean_std_str)
    else:
        mean_std_dict = mean_std_str

    # Get EEG data and channel names
    data = raw_data.get_data()
    channel_names = raw_data.info['ch_names']

    if norm_type == "channel-wise":
        # Normalize each channel based on its individual mean and std
        for idx, channel in enumerate(channel_names):
            if channel in mean_std_dict:
                mean, std = mean_std_dict[channel]
                data[idx] = (data[idx] - mean) / std

    elif norm_type == "sample-wise":
        # Normalize all channels based on the common mean and std
        mean, std = mean_std_dict.get('all_eeg', (None, None))
        if mean is None or std is None:
            raise ValueError("Mean and std for 'all_eeg' are required for sample-wise normalization.")
        for idx in range(data.shape[0]):
            data[idx] = (data[idx] - mean) / std

    else:
        raise ValueError(f"Invalid norm_type: {norm_type}. Must be 'channel-wise' or 'sample-wise'.")

    # Set the normalized data back to raw_data
    raw_data._data = data

    return raw_data


def set_montage_any(raw_data: mne.io.Raw, verbose='CRITICAL'):
    """
    Set the montage for the given raw data using a montage defined in a JSON file.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw data object to which the montage will be applied.
    verbose : str, optional
        The verbosity level for warnings or messages, by default 'CRITICAL'.

    Returns
    -------
    mne.io.Raw
        The updated raw data object with the applied montage.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    montage = create_montage_from_json(os.path.join(current_dir, 'combined_montage.json'))
    raw_data.set_montage(montage, on_missing='warn', verbose=verbose)
    return raw_data


def create_montage_from_json(json_file):
    """
    Create a montage from a JSON file containing channel positions.

    Parameters
    ----------
    json_file : str
        The path to the JSON file containing channel names as keys and their positions as values.

    Returns
    -------
    mne.channels.DigMontage
        A montage object created from the channel positions defined in the JSON file.
    """
    with open(json_file, 'r') as f:
        montage_data = json.load(f)

    ch_names = list(montage_data.keys())
    pos = [montage_data[ch_name] for ch_name in ch_names]

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, pos)))

    return montage


def set_channel_type(raw_data, channel_str):
    """
    Set the channel types for the given raw data based on the specified channel string.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw data object containing the EEG, EMG, ECG, EOG, or other types of signals.
    channel_str : str
        A string specifying the channel types and names in the format 'type:name', separated by commas. Each type must correspond to the desired signal type.

    Returns
    -------
    mne.io.Raw
        The updated raw data object with renamed channels and set channel types.

    Raises
    ------
    ValueError
        If the format of any channel in the channel string is invalid (not in 'type:name' format).
    """
    channel_info = [ch.split(':') for ch in channel_str.split(',')]
    for ch in channel_info:
        if len(ch) != 2:
            raise ValueError(f"Invalid channel format: {ch}. Each channel must be in 'type:name' format.")

    channel_mapping = {raw_channel: new_channel for raw_channel, new_channel in
                       zip(raw_data.info['ch_names'], [ch[1] for ch in channel_info])}
    raw_data.rename_channels(channel_mapping)
    for ch, ch_info in zip(raw_data.info['chs'], channel_info):
        if ch_info[0].strip() == 'EEG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_EEG_CH
        elif ch_info[0].strip() == 'EMG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_EMG_CH
        elif ch_info[0].strip() == 'ECG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_ECG_CH
        elif ch_info[0].strip() == 'EOG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_EOG_CH
        elif ch_info[0].strip() == 'STIM':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_STIM_CH
        else:
            ch['kind'] = mne.io.constants.FIFF.FIFFV_BIO_CH
    return raw_data


def get_data_row(row: dict,
                 norm_type: str = None,
                 is_set_channel_type: bool = False,
                 is_set_montage: bool = False,
                 verbose: str = 'CRITICAL',
                 pick_types: dict = None,
                 unit_convert: str = None,
                 **kwargs) -> mne.io.BaseRaw:
    """
    Process and return raw EEG data based on the input row information.
    Allows handling of standard and non-standard data with options for
    setting channel types, montage, normalization, and unit conversion.

    Parameters
    ----------
    row : dict
        Dictionary containing data attributes, such as file paths, file types,
        and channel names.
    norm_type : str, optional
        Type of normalization to apply, if any. Defaults to None.
    is_set_channel_type : bool, optional
        Whether to set channel types based on provided information. Defaults to False.
    is_set_montage : bool, optional
        Whether to set montage (electrode coordinates). Defaults to False.
    verbose : str, optional
        Verbosity level for MNE functions. Defaults to 'CRITICAL'.
    pick_types : dict, optional
        Dictionary specifying which channel types to include. The keys should
        match the parameters of `raw.pick_types()`. Defaults to None.
    unit_convert : str, optional
        Conversion type for resetting channel units. Defaults to None.
    kwargs : any
        Additional keyword arguments to enhance flexibility and robustness.

    Returns
    -------
    mne.io.BaseRaw
        The processed raw EEG data object.

    Raises
    ------
    ValueError
        If the number of channels in the locator file does not match the metadata.
    Warning
        If `pick_types` is not None but `is_set_channel_type` is False, a warning
        will be issued to inform the user to set `is_set_channel_type=True`.
    """
    filepath = row.get('File Path')
    file_type = row.get('File Type')

    # Handle standard or non-standard data loading based on file type
    if file_type == "standard_data":  # Read standard data supported by MNE-Python
        raw_data = mne.io.read_raw(filepath, verbose=verbose, preload=True)
        channel_names = [name.strip() for name in row.get('Channel Names', '').split(',')]
        if len(channel_names) != len(raw_data.info['ch_names']):
            raise ValueError(f"The number of channels marked in the locator file does not match the metadata: {filepath}")
        channel_mapping = {original: new for original, new in zip(raw_data.info['ch_names'], channel_names)}
        raw_data.rename_channels(channel_mapping)
    else:  # Handle non-standard data
        raw_data = handle_nonstandard_data(row, verbose=verbose, **kwargs)

    # Check if pick_types is provided without is_set_channel_type being True
    if pick_types is not None and not is_set_channel_type:
        warnings.warn("When `pick_types` is not None, it's recommended to set `is_set_channel_type=True`.")

    # Set channel types if required
    if is_set_channel_type:
        raw_data = set_channel_type(raw_data, row['Channel Names'])

        # Apply pick types if provided
        if pick_types is not None:
            raw_data = raw_data.pick_types(**pick_types)  # Unpack the dictionary here

    # Set montage if required
    if is_set_montage:
        raw_data = set_montage_any(raw_data)

    # Apply normalization if requested
    if norm_type and 'MEAN STD' in row:
        raw_data = normalize_data(raw_data, row['MEAN STD'], norm_type)

    # Convert units if required
    if unit_convert and 'Infer Unit' in row:
        raw_data = set_infer_unit(raw_data, row)
        raw_data = convert_unit(raw_data, unit_convert)

    # Check and reset incorrect timestamps if needed
    if raw_data.info['meas_date'] is not None and isinstance(raw_data.info['meas_date'], datetime.datetime):
        timestamp = raw_data.info['meas_date'].timestamp()
        if timestamp < -2147483648 or timestamp > 2147483647:
            raw_data.set_meas_date(None)

    return raw_data


def set_infer_unit(raw_data, row):
    """
    Set the inferred unit for EEG channels in the raw data.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw data object containing the EEG channels.

    row : pandas.Series
        A row from a DataFrame containing the 'Infer Unit' field, which should be a dictionary with channel names as keys and units as values.

    Returns
    -------
    mne.io.Raw
        The updated raw data object with the inferred units set for the specified channels.

    Raises
    ------
    ValueError
        If 'Infer Unit' is not a valid dictionary.
    """
    infer_unit = ast.literal_eval(row['Infer Unit'])
    if isinstance(infer_unit, dict):
        for ch_name, unit in infer_unit.items():
            if ch_name in raw_data.info['ch_names']:
                idx = raw_data.ch_names.index(ch_name)
                raw_data.info['chs'][idx]['eegunity_unit'] = unit
        return raw_data
    else:
        raise ValueError(f"'Infer Unit' is not a valid dictionary: {row['Infer Unit']}")


def channel_name_parser(input_string):
    """
    Format and standardize a list of channel names based on predefined rules.

    Parameters
    ----------
    input_string : str
        A comma-separated string containing channel names to be formatted.

    Returns
    -------
    str
        A comma-separated string of formatted channel names. If duplicates are found, the original input string is returned.

    Warnings
    --------
    Warns if an invalid channel name is detected or if a duplicate formatted channel name is found.
    """
    # Define a function to check if a channel is an EOG channel
    def is_eog_channel(channel):
        return "eog" in channel.lower()

    # Define a function to check if a channel is an MEG channel
    def is_meg_channel(channel):
        return "meg" in channel.lower()

    # Define a function to check if a channel is an ECG channel
    def is_ecg_channel(channel):
        return "ecg" in channel.lower()

    # Define a function to check if a channel is an ECG channel
    def is_stim_channel(channel):
        return "stim" in channel.lower()

    # Define a function to check if a channel is an DOUBLE channel
    def is_double_channel(channel):
        if '-' not in channel:
            return False
        parts = channel.split('-')
        if len(parts) != 2:
            return False

        valid_channels = ["REF", "LE", "EEG", "ECG", "EOG", "EMG"]

        # Convert both parts to upper case for case-insensitive comparison
        part1, part2 = parts[0].upper(), parts[1].upper()

        if part1 in valid_channels or part2 in valid_channels:
            return False
        return True

    # Define a function to standardize channel names
    def standardize_eeg_channel(channel):
        # remove prefix
        channel = channel.replace('EEG:', '').replace('EEG', '').replace('eeg', '')
        return f"EEG:{channel}"

    def remap_standard_name(channel):
        # remove prefix
        channel = channel.replace('EEG:', '').replace('EEG', '').replace('eeg', '')

        # define rules for replacement
        replacements = OrderedDict([
            ('FAF', 'AFF'),
            ('CFC', 'FCC'),
            ('CPC', 'CCP'),
            ('POP', 'PPO'),
            ('TPT', 'TTP'),
            ('TFT', 'FTT')
        ])
        # employ replacement and propose warnings
        for old, new in replacements.items():
            if old.lower() in channel.lower():
                warnings.warn(
                    f'{old.upper()} is an invalid 10-5 name and has been replaced with {new.upper()}. \n If mismatch happen, you should change locator manually.')
                channel = channel.lower().replace(old.lower(), new.lower())
        return channel

    # Define a function to preprocess channel names by removing leading/trailing whitespace and EEG-related prefixes/suffixes
    def preprocess_channel(channel):
        for prefix_suffix in EEG_PREFIXES_SUFFIXES:
            if channel.startswith(prefix_suffix + "-"):
                channel = channel[len(prefix_suffix + "-"):].strip()
            if channel.startswith(prefix_suffix + ":"):
                channel = channel[len(prefix_suffix + ":"):].strip()
            if channel.startswith(prefix_suffix):
                channel = channel[len(prefix_suffix):].strip()
            if channel.endswith("-" + prefix_suffix):
                channel = channel[:-len("-" + prefix_suffix)].strip()
            if channel.endswith(prefix_suffix):
                channel = channel[:-len(prefix_suffix)].strip()
        return channel

    # Split the input string into a list of channel names
    channels = [channel.strip() for channel in input_string.split(',')]

    # Initialize a set for formatted channel names and a set for seen channel names
    formatted_channels = []
    seen_channels = set()

    # Process each channel name
    for channel in channels:
        channel = preprocess_channel(channel)

        if is_eog_channel(channel):
            formatted_channel = f"EOG:{channel.replace('EOG:', '').replace('EOG', '').replace('eog', '')}"
        elif is_meg_channel(channel):
            formatted_channel = f"MEG:{channel.replace('MEG:', '').replace('MEG', '').replace('meg', '')}"
        elif is_ecg_channel(channel):
            formatted_channel = f"ECG:{channel.replace('ECG:', '').replace('ECG', '').replace('ecg', '')}"
        elif is_double_channel(channel):
            formatted_channel = f"EEGDual:{channel.replace('Dual:', '')}"
        elif is_stim_channel(channel):
            formatted_channel = f"STIM:{channel.replace('STIM:', '')}"
        else:
            re_channel = remap_standard_name(channel)
            matched = False
            for standard_name in STANDARD_EEG_CHANNELS:
                if re_channel.lower() == standard_name.lower():
                    formatted_channel = standardize_eeg_channel(standard_name)
                    matched = True
                    break
            if not matched:
                formatted_channel = f"Unknown:{channel.replace('Unknown:', '')}"
        # Check for duplicates
        if formatted_channel in seen_channels:
            warnings.warn(f"Duplicate formatted channel detected: {formatted_channel}")
            return input_string

        formatted_channels.append(formatted_channel)
        seen_channels.add(formatted_channel)

    # Concatenate the formatted channel names into a string
    output_string = ', '.join(formatted_channels)
    return output_string


def _clean_sampling_rate_(df):
    """
    Cleans the 'Sampling Rate' column in the provided DataFrame by removing invalid characters.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the 'Sampling Rate' column to be cleaned.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the cleaned 'Sampling Rate' column.
    """
    # Convert the 'Sampling Rate' column to a string and retain only digits, decimal points, and the 'e' in scientific notation
    df['Sampling Rate'] = df['Sampling Rate'].astype(str).apply(lambda x: re.sub(r'[^0-9.eE+-]', '', x))
    # If necessary, convert the result back to a numeric type
    df['Sampling Rate'] = pd.to_numeric(df['Sampling Rate'], errors='coerce')

    return df


def handle_nonstandard_data(row, verbose='CRITICAL'):
    """
    Handles the loading of non-standard EEG data files into MNE Raw format.

    This function processes EEG data from either .mat files or .csv/.txt files marked as 'csvData'.
    It extracts channel names and sampling rates from the provided row, and creates an MNE RawArray object containing the EEG data.

    Parameters
    ----------
    row : pd.Series
        A row from a DataFrame containing information about the file, including 'File Path', 'Channel Names', 'Sampling Rate', and 'File Type'.

    verbose : str, optional
        The verbosity level for MNE functions. Default is 'CRITICAL'.

    Returns
    -------
    mne.io.Raw
        An MNE Raw object containing the EEG data.

    Raises
    ------
    ValueError
        If the number of channels in the DataFrame does not match the channel names specified in the row.
    Exception
        If the file type is unsupported or there is an error in loading the data.
    """
    filepath = row['File Path']
    if filepath.endswith('.mat'):
        matdata = scipy.io.loadmat(filepath)
        eeg_data = \
            _find_variables_by_condition(matdata, _condition_source_data, max_depth=5, max_width=20)[1]
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        channel_names = row['Channel Names'].split(',')
        info = mne.create_info(ch_names=channel_names, sfreq=float(row['Sampling Rate']), ch_types='eeg',
                               verbose=verbose)
        raw = mne.io.RawArray(eeg_data, info)
        return raw

    elif (filepath.endswith('.csv') or filepath.endswith('.txt')) and row['File Type'] == 'csvData':
        # Retrieve header information from the locator
        # Check if there is a header
        header_option = None if row.get('Header', 'infer') == 'None' else 'infer'
        df = pd.read_csv(filepath, header=header_option)

        if header_option is None:
            # If the file has no column names, generate them
            df.columns = [str(i) for i in range(1, len(df.columns) + 1)]

        # Retrieve channel names and sampling rate
        channel_names = [name.strip() for name in row['Channel Names'].split(',')]
        sfreq = float(row['Sampling Rate'])

        # Check if all channel names are present in the DataFrame columns
        if not all(name in df.columns for name in channel_names):
            raise (f"Number of channels marked in the locator file does not match the metadata channels {filepath}")
        # Extract EEG data
        eeg_data = df[channel_names].values.T  # transpose for MNE
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
        return raw

    else:
        raise ("Parsing of files other than .mat/.csv/.txt is currently not supported")


def extract_events(raw):
    """
    Extract events from an mne.io.Raw object.

    Attempt to extract events using mne.find_events.
    If it fails, use mne.events_from_annotations to extract events.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object.

    Returns
    -------
    events : numpy.ndarray
        The events array, shaped (n_events, 3).
    event_id : dict
        Dictionary of event IDs.
    """
    try:
        # Attempt to extract events using mne.find_events
        events = mne.find_events(raw)
        # Automatically generate the event_id dictionary, assuming all events are valid
        unique_event_ids = np.unique(events[:, 2])
        event_id = {f'event_{event_id}': event_id for event_id in unique_event_ids}
    except ValueError as e:
        # Extract events using mne.events_from_annotations
        events, event_id = mne.events_from_annotations(raw)
    return events, event_id


def infer_channel_unit(ch_name, ch_data, ch_type):
    """
    Infer the unit type for a given channel based on its data and type.

    Parameters
    ----------
    ch_name : str
        The name of the channel.
    ch_data : array-like
        The data of the channel, typically an array of amplitude values.
    ch_type : str
        The type of the channel, such as 'eeg', 'emg', etc.

    Returns
    -------
    str
        The inferred unit type, such as "uV", "mV", or "V", based on the channel data and type.
    """

    mean_val = abs(ch_data).mean()

    # Infer the unit based on the channel type and average amplitude
    if ch_type == 'eeg' or ch_type == 'eog':
        if mean_val > 1:
            return "uV"
        elif mean_val > 0.001:
            return "mV"
        else:
            return "V"
    elif ch_type == 'ecg' or ch_type == 'emg':
        if mean_val > 1:
            return "uV"
        elif mean_val > 0.001:
            return "mV"
        else:
            return "V"
    else:
        # For misc and other unknown types
        if mean_val > 1:
            return "uV"
        elif mean_val > 0.001:
            return "mV"
        else:
            return "V"


def convert_unit(data: mne.io.Raw, unit: str) -> mne.io.Raw:
    """
    Convert the units of EEG data in a MNE Raw object.

    Parameters
    ----------
    data : mne.io.Raw
        The raw EEG data to be converted.
    unit : str
        The target unit to convert the data to. Must be one of 'V', 'mV', or 'uV'.

    Raises
    ------
    ValueError
        If the provided unit is not valid.

    Returns
    -------
    mne.io.Raw
        The raw EEG data with converted units.
    """
    # Validate the unit
    valid_units = ['V', 'mV', 'uV']
    if unit not in valid_units:
        raise ValueError(f"Invalid unit '{unit}'. Valid units are 'V', 'mV', 'uV'.")

    # Get the number of channels
    n_channels = len(data.info['chs'])

    # Define unit conversion relationships
    unit_conversion = {'V': 1, 'mV': 1e-3, 'uV': 1e-6}
    target_multiplier = unit_conversion[unit]

    # Iterate through all channels
    for i in range(n_channels):
        ch = data.info['chs'][i]
        current_unit = ch['eegunity_unit']

        if current_unit in unit_conversion:
            current_multiplier = unit_conversion[current_unit]
            conversion_factor = current_multiplier / target_multiplier
            # Convert the data
            data._data[i] *= conversion_factor

            # Update the unit
            ch['eegunity'] = unit

        # Mark that the unit conversion has been done
        data.info['chs'][i].update({"eegunity_unit_converted": unit})

    return data


def process_mne_files(files_locator, verbose):
    """
    Process MNE files based on a locator DataFrame.

    Parameters
    ----------
    files_locator : pandas.DataFrame
        DataFrame containing file paths and related metadata for processing.
    verbose : str
        Verbosity level for MNE functions.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with metadata extracted from processed files.
    """
    for index, row in files_locator.iterrows():
        filepath = row['File Path']
        try:
            data = mne.io.read_raw(filepath, verbose=verbose)
            files_locator.at[index, 'File Type'] = 'standard_data'
            files_locator.at[index, 'Data Shape'] = str((data.info['nchan'], data.n_times))
            files_locator.at[index, 'Channel Names'] = ', '.join(data.info['ch_names'])
            files_locator.at[index, 'Number of Channels'] = len(data.info['ch_names'])
            files_locator.at[index, 'Sampling Rate'] = data.info['sfreq']
            files_locator.at[index, 'Duration'] = data.times[-1]
            print(f"Retrieved channel sequence {data.info['ch_names']}")
        except Exception:
            files_locator.at[index, 'File Type'] = 'unknown'
    return files_locator
