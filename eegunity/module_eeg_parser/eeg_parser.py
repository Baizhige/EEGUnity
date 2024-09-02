import os
import pandas as pd
import glob
import mne
import zipfile
import re
import scipy
from eegunity.share_attributes import UDatasetSharedAttributes
from eegunity.module_eeg_parser.eeg_parser_mat import process_mat_files, _find_variables_by_condition, _condition_source_data
from eegunity.module_eeg_parser.eeg_parser_csv import process_csv_files
import ast
import warnings
import json
import datetime
import numpy as np


current_dir = os.path.dirname(__file__)
json_file_path = os.path.join(current_dir, 'combined_montage.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)
STANDARD_EEG_CHANNELS = list(data.keys())
EEG_PREFIXES_SUFFIXES = {"EEG", "FP", "REF", "LE", "RE"}

class EEGParser(UDatasetSharedAttributes):
    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr
        self.format_channel_names = format_channel_names
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
                raise ValueError("The provided 'locator' path is not a valid CSV file.")
        elif self.get_shared_attr()['dataset_path']:  # Construct UnifiedDataset by reading dataset path
            if os.path.isdir(dataset_path):
                self._unzip_if_no_conflict(dataset_path)
                self.set_shared_attr({'locator': self.check_locator(self._process_directory(dataset_path))})
            else:
                raise ValueError("The provided 'datasets' path is not a valid directory.")

    def _process_directory(self, datasets_path):
        files_info = []
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
        return files_locator

    def _unzip_if_no_conflict(self, datasets_path):
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

    def get_data(self, data_idx, norm_type=None, unit_convert=False):
        row = self.get_shared_attr()['locator'].iloc[data_idx]
        return get_data_row(row, norm_type, self.get_shared_attr()['verbose'], unit_convert=unit_convert)

    def check_locator(self, locator):
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
def normalize_data(raw_data, mean_std_str, norm_type):
    mean_std_str = mean_std_str.replace('nan', 'None')
    mean_std_dict = ast.literal_eval(mean_std_str)
    # get content data
    data = raw_data.get_data()
    channel_names = raw_data.info['ch_names']
    if norm_type == "channel-wise":
        for idx, channel in enumerate(channel_names):
            if channel in mean_std_dict:
                mean, std = mean_std_dict[channel]
                data[idx] = (data[idx] - mean) / std

    elif norm_type == "sample-wise":
        mean, std = mean_std_dict['all_eeg']
        for idx in range(data.shape[0]):
            data[idx] = (data[idx] - mean) / std

    # return mne.io.raw data
    raw_data._data = data
    return raw_data


def set_montage_any(raw_data: mne.io.Raw, verbose='CRITICAL'):
    montage = create_montage_from_json('combined_montage.json')
    raw_data.set_montage(montage, on_missing= 'warn', verbose=verbose)
    return raw_data


def create_montage_from_json(json_file):
    with open(json_file, 'r') as f:
        montage_data = json.load(f)

    ch_names = list(montage_data.keys())
    pos = [montage_data[ch_name] for ch_name in ch_names]

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, pos)))

    return montage

def set_channel_type(raw_data, channel_str):
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
        else:
            ch['kind'] = mne.io.constants.FIFF.FIFFV_BIO_CH
    return raw_data

def get_data_row(row, norm_type=None, is_set_channel_type=True, is_set_montage=True, verbose='CRITICAL', pick_types=None, unit_convert=None):
    filepath = row['File Path']
    file_type = row['File Type']
    # get mne.io.raw data
    if file_type == "standard_data": # read standard data, those supported by MNE-Python
        raw_data = mne.io.read_raw(filepath, verbose=verbose, preload=True)
        channel_names = [name.strip() for name in row['Channel Names'].split(',')]
        if len(channel_names) != len(raw_data.info['ch_names']):
            raise ValueError(f"The number of channels marked in the locator file does not match the number of channels in the metadata: {filepath}")
        channel_mapping = {original: new for original, new in zip(raw_data.info['ch_names'], channel_names)}
        raw_data.rename_channels(channel_mapping)
    else: # cope with non-standard data
        raw_data = handle_nonstandard_data(row, verbose)
    # Reset channel names and types based on the locator
    if is_set_channel_type:
        raw_data = set_channel_type(raw_data, row['Channel Names'])
    # Set electrode coordinates according to the preset montage
    if is_set_montage:
        raw_data = set_montage_any(raw_data)
    # Apply normalization
    if norm_type and 'MEAN STD' in row:
        raw_data = normalize_data(raw_data, row['MEAN STD'], norm_type)
    # Reset channel units
    if unit_convert and 'Infer Unit' in row:
        raw_data = set_infer_unit(raw_data, row)
        raw_data = convert_unit(raw_data, unit_convert)
    # Reset when there are timestamp anomalies
    if raw_data.info['meas_date'] is not None and isinstance(raw_data.info['meas_date'],
                                                            datetime.datetime) and (
            raw_data.info['meas_date'].timestamp() < -2147483648 or raw_data.info[
        'meas_date'].timestamp() > 2147483647):
        # If the date is wrong, set None.
        raw_data.set_meas_date(None)
    return raw_data

def set_infer_unit(raw_data, row):
    infer_unit = ast.literal_eval(row['Infer Unit'])
    if isinstance(infer_unit, dict):
        for ch_name, unit in infer_unit.items():
            if ch_name in raw_data.info['ch_names']:
                idx = raw_data.ch_names.index(ch_name)
                raw_data.info['chs'][idx]['eegunity_unit'] = unit
        return raw_data
    else:
        raise ValueError(f"'Infer Unit' is not a valid dictionary: {row['Infer Unit']}")
def format_channel_names(input_string):

    # Define a function to check if a channel is an EOG channel
    def is_eog_channel(channel):
        return "eog" in channel.lower()

    # Define a function to check if a channel is an MEG channel
    def is_meg_channel(channel):
        return "meg" in channel.lower()

    # Define a function to check if a channel is an ECG channel
    def is_ecg_channel(channel):
        return "ecg" in channel.lower()

    # Define a function to check if a channel is an DOUBLE channel
    def is_double_channel(channel):
        if '-' not in channel:
            return False
        parts = channel.split('-')
        if len(parts) != 2:
            return False

        valid_channels = {"REF", "LE", "EEG", "ECG", "EOG", "EMG"}

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
        replacements = {
            'FAF': 'AFF',
            'CFC': 'FCC',
            'CPC': 'CCP',
            'POP': 'PPO',
            'TPT': 'TTP',
            'TFT': 'FTT'
        }
        # employ replacement and propose warnings
        for old, new in replacements.items():
            if old.lower() in channel.lower():
                warnings.warn(
                    f'{old.upper()} is an invalid 10-5 name and has been replaced with {new.upper()}. \n If mismatch happen, you should change locator manually.')
                channel = channel.lower().replace(old.lower(), new.lower())
        return channel

    # Define a function to preprocess channel names by removing leading/trailing whitespace and EEG-related prefixes/suffixes
    def preprocess_channel(channel):
        channel = channel.replace(" ", "")
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
    channels = input_string.split(',')

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

    # Convert the 'Sampling Rate' column to a string and retain only digits, decimal points, and the 'e' in scientific notation
    df['Sampling Rate'] = df['Sampling Rate'].astype(str).apply(lambda x: re.sub(r'[^0-9.eE+-]', '', x))
    # If necessary, convert the result back to a numeric type
    df['Sampling Rate'] = pd.to_numeric(df['Sampling Rate'], errors='coerce')

    return df

def handle_nonstandard_data(row, verbose='CRITICAL'):
    filepath = row['File Path']
    if filepath.endswith('.mat'):
        matdata = scipy.io.loadmat(filepath)
        eeg_data = \
            _find_variables_by_condition(matdata, _condition_source_data, max_depth=5, max_width=20)[1]
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        channel_names = row['Channel Names'].split(',')
        info = mne.create_info(ch_names=channel_names, sfreq=float(row['Sampling Rate']), ch_types='eeg',verbose=verbose)
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
        channel_names = row['Channel Names'].split(',')
        sfreq = float(row['Sampling Rate'])

        # Check if all channel names are present in the DataFrame columns
        if not all(name in df.columns for name in channel_names):
            raise(f"Number of channels marked in the locator file does not match the metadata channels {filepath}")
        # Extract EEG data
        eeg_data = df[channel_names].values.T  # 转置以匹配 MNE 需要的格式 (通道数, 时间点数)
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
        return raw

    else:
        raise("Parsing of files other than .mat/.csv/.txt is currently not supported")

def extract_events(raw):
    """
    Extract events from an mne.io.Raw object.

    Attempt to extract events using mne.find_events.
    If it fails, use mne.events_from_annotations to extract events.

    Parameters:
    raw : mne.io.Raw
        The raw data object.

    Returns:
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
    Infer the unit type for a channel.

    Parameters:
    ch_name: The name of the channel
    ch_data: The data of the channel
    ch_type: The type of the channel

    Returns:
    The inferred unit type, such as "uV", "mV", "V", etc.
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