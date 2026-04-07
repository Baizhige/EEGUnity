import ast
import datetime
import glob
import json
import os
import re
import warnings
import zipfile
import functools
import inspect
import mne
import numpy as np
import pandas as pd
import scipy
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict, Tuple, Optional, List
from collections import OrderedDict
from eegunity.modules.parser.eeg_parser_csv import process_csv_files
from eegunity.modules.parser.eeg_parser_mat import (
    process_mat_files, process_hdf5_set_files, read_eeglab_hdf5,
    _find_variables_by_condition, _condition_source_data,
)
from eegunity.modules.parser.eeg_parser_wfdb import process_wfdb_files
from eegunity._share_attributes import _UDatasetSharedAttributes


def _apply_mne_fieldtrip_verbose_patch():
    """Patch ``mne.io.read_raw_fieldtrip`` to ignore unknown kwargs.

    Some MNE versions pass ``verbose`` to readers that do not declare it,
    which can break FieldTrip loading. This patch wraps the FieldTrip reader
    and filters unsupported keyword arguments.

    Examples
    --------
    >>> _apply_mne_fieldtrip_verbose_patch()  # doctest: +SKIP
    """
    try:
        original_reader = mne.io.read_raw_fieldtrip
        if 'verbose' in inspect.signature(original_reader).parameters:
            return
        valid_params = set(inspect.signature(original_reader).parameters)

        @functools.wraps(original_reader)
        def _patched_reader(*args, **kwargs):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return original_reader(*args, **filtered_kwargs)

        mne.io.read_raw_fieldtrip = _patched_reader

        # MNE >= 1.x may cache function objects in the internal dispatch table.
        try:
            from mne.io._read_raw import _get_supported
            supported = _get_supported()
            for readers in supported.values():
                if isinstance(readers, dict) and 'fieldtrip' in readers:
                    readers['fieldtrip'] = _patched_reader
        except Exception:
            pass
    except Exception:
        pass


_apply_mne_fieldtrip_verbose_patch()

current_dir = os.path.dirname(__file__)
json_file_path = os.path.join(current_dir, '..', '..', 'resources','combined_montage.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)
STANDARD_EEG_CHANNELS = sorted(data.keys(), key=len, reverse=True)
EEG_PREFIXES_SUFFIXES = ["EEG", "REF", "LE", "-", "_", ":", "."]

# Locator prefix aliases. Values are MNE-compatible channel type strings.
_CHANNEL_TYPE_ALIASES = {
    'EEG': 'eeg',
    'EOG': 'eog',
    'EMG': 'emg',
    'ECG': 'ecg',
    'EKG': 'ecg',
    'STIM': 'stim',
    'MISC': 'misc',
    'UNKNOWN': 'bio',
    'BIO': 'bio',
    # Legacy EEGUnity prefix kept for backward compatibility.
    'MEG': 'meg',
    'MAG': 'mag',
    'GRAD': 'grad',
    'REF_MEG': 'ref_meg',
    'RESP': 'resp',
    'DBS': 'dbs',
    'SEEG': 'seeg',
    'ECOG': 'ecog',
    'TEMP': 'temperature',
    'TEMPERATURE': 'temperature',
    'GSR': 'gsr',
    'PUPIL': 'pupil',
    'EYETRACK': 'eyetrack',
    'EYEGAZE': 'eyegaze',
    'FNIRS': 'fnirs',
    'CSD': 'csd',
    'HBO': 'hbo',
    'HBR': 'hbr',
    'FNIRS_OD': 'fnirs_od',
    'FNIRS_CW_AMPLITUDE': 'fnirs_cw_amplitude',
    'FNIRS_FD_AC_AMPLITUDE': 'fnirs_fd_ac_amplitude',
    'FNIRS_FD_PHASE': 'fnirs_fd_phase',
    'CHPI': 'chpi',
    'DIPOLE': 'dipole',
    'GOF': 'gof',
    'EXCI': 'exci',
    'IAS': 'ias',
    'SYST': 'syst',
}


@functools.lru_cache(maxsize=256)
def _resolve_mne_channel_type(channel_type: str) -> Optional[str]:
    """Resolve a locator channel type string to an MNE channel type.

    Parameters
    ----------
    channel_type : str
        Raw channel type prefix from locator (for example ``"EEG"`` or
        ``"seeg"``).

    Returns
    -------
    str or None
        Canonical MNE channel type string if resolvable, otherwise ``None``.

    Examples
    --------
    >>> _resolve_mne_channel_type("EEG")
    'eeg'
    >>> _resolve_mne_channel_type("seeg")
    'seeg'
    """
    raw_type = str(channel_type).strip()
    if not raw_type:
        return None

    candidate = _CHANNEL_TYPE_ALIASES.get(raw_type.upper(), raw_type.lower())
    try:
        # Validation against the installed MNE type registry.
        mne.create_info(['_tmp_'], sfreq=1.0, ch_types=[candidate])
        return candidate
    except Exception:
        # Backward/forward compatibility fallbacks across MNE versions.
        fallback_map = {
            'meg': 'mag',
            'eyetrack': 'eyegaze',
            'fnirs': 'fnirs_cw_amplitude',
        }
        fallback = fallback_map.get(candidate)
        if fallback is not None:
            try:
                mne.create_info(['_tmp_'], sfreq=1.0, ch_types=[fallback])
                return fallback
            except Exception:
                return None
        return None


@functools.lru_cache(maxsize=256)
def _channel_type_template(channel_type: str) -> Dict[str, int]:
    """Return channel-info template fields for a given MNE channel type.

    Parameters
    ----------
    channel_type : str
        Canonical MNE channel type string.

    Returns
    -------
    dict
        Dictionary with keys ``kind``, ``coil_type``, ``unit``,
        ``unit_mul``, and ``coord_frame``.

    Examples
    --------
    >>> tpl = _channel_type_template("eeg")
    >>> isinstance(tpl["kind"], int)
    True
    """
    try:
        ch = mne.create_info(['_tmp_'], sfreq=1.0, ch_types=[channel_type])['chs'][0]
    except Exception:
        ch = mne.create_info(['_tmp_'], sfreq=1.0, ch_types=['bio'])['chs'][0]
    return {
        'kind': ch['kind'],
        'coil_type': ch['coil_type'],
        'unit': ch['unit'],
        'unit_mul': ch.get('unit_mul', 0),
        'coord_frame': ch.get('coord_frame', 0),
    }


def _split_typed_channel(channel_entry: str) -> Tuple[str, str]:
    """Split one locator channel entry into ``(type, name)``.

    Parameters
    ----------
    channel_entry : str
        Single channel descriptor in ``"type:name"`` format.

    Returns
    -------
    tuple of str
        ``(channel_type, channel_name)``.

    Raises
    ------
    ValueError
        If ``channel_entry`` is not in ``"type:name"`` format.

    Examples
    --------
    >>> _split_typed_channel("eeg:Fz")
    ('eeg', 'Fz')
    """
    if ':' not in channel_entry:
        raise ValueError(
            f"Invalid channel format: {channel_entry}. Each channel must be in 'type:name' format."
        )
    ch_type, ch_name = channel_entry.split(':', 1)
    ch_type = ch_type.strip()
    ch_name = ch_name.strip()
    if not ch_type or not ch_name:
        raise ValueError(
            f"Invalid channel format: {channel_entry}. Each channel must be in 'type:name' format."
        )
    return ch_type, ch_name


def _locator_prefix_from_mne_type(original_prefix: str, mne_type: str) -> str:
    """Choose a locator prefix for a resolved MNE channel type.

    Parameters
    ----------
    original_prefix : str
        Original prefix found in locator.
    mne_type : str
        Canonical MNE channel type string.

    Returns
    -------
    str
        Prefix written back to locator-style ``type:name`` strings.

    Examples
    --------
    >>> _locator_prefix_from_mne_type("eeg", "eeg")
    'eeg'
    >>> _locator_prefix_from_mne_type("seeg", "seeg")
    'seeg'
    """
    upper = original_prefix.strip().upper()
    if upper in {'EEG', 'EOG', 'EMG', 'ECG', 'EKG', 'STIM', 'MISC', 'BIO'}:
        if upper == 'EKG':
            return 'ecg'
        return upper.lower()
    if upper == 'UNKNOWN':
        return 'bio'
    if upper == 'MEG':
        return 'meg'
    if upper in {'MAG', 'GRAD'}:
        return upper.lower()
    if mne_type in {'mag', 'grad'}:
        return 'meg'
    return mne_type


def _is_typed_channel_string(channel_string: str) -> bool:
    """Return ``True`` when all channel entries are ``type:name`` pairs.

    Parameters
    ----------
    channel_string : str
        Comma-separated locator channel string.

    Returns
    -------
    bool
        ``True`` if every channel entry contains a non-empty type and name.

    Examples
    --------
    >>> _is_typed_channel_string("eeg:Fz, eog:LOC")
    True
    >>> _is_typed_channel_string("Fz, LOC")
    False
    """
    entries = [entry.strip() for entry in str(channel_string).split(',') if entry.strip()]
    if not entries:
        return False
    for entry in entries:
        try:
            _split_typed_channel(entry)
        except ValueError:
            return False
    return True


def apply_dataset_kernel(udataset, raw_data: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
    """Apply the dataset kernel to one loaded raw object.

    Parameters
    ----------
    udataset : object
        UnifiedDataset-like object exposing ``get_shared_attr()``.
    raw_data : mne.io.BaseRaw
        Loaded raw object after locator-driven metadata patching.
    row : pandas.Series
        Locator row corresponding to ``raw_data``.

    Returns
    -------
    mne.io.BaseRaw
        Kernel-processed raw object, or the original object if no kernel is
        bound or if kernel execution fails.

    Examples
    --------
    >>> # raw = apply_dataset_kernel(unified_dataset, raw, row)  # doctest: +SKIP
    """
    if udataset is None:
        return raw_data

    try:
        shared_attr = udataset.get_shared_attr()
    except Exception:
        return raw_data

    kernel = shared_attr.get('kernel', None)
    if kernel is None:
        return raw_data

    try:
        return kernel.apply(udataset, raw_data, row)
    except Exception as e:
        kid = getattr(kernel, "KERNEL_ID", kernel.__class__.__name__)
        domain_tag = row.get("Domain Tag", "unknown")
        file_path = row.get("File Path", "unknown")
        warnings.warn(
            (
                f"Kernel '{kid}' is not compatible with this dataset (or this record). "
                f"Domain Tag: {domain_tag}. File Path: {file_path}. "
                f"Error: {e}. Please adjust the kernel or download a dataset version "
                f"that matches the kernel."
            )
        )
        return raw_data


class EEGParser(_UDatasetSharedAttributes):
    def __init__(self, main_instance):
        super().__init__()
        self.main_instance = main_instance
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
        num_workers = self.get_shared_attr().get('num_workers', 0)
        min_file_size = self.get_shared_attr().get('min_file_size', 5 * 1024 * 1024)
        files_locator = process_mne_files(files_locator, self.get_shared_attr()['verbose'], num_workers=num_workers)
        files_locator = process_mat_files(files_locator, num_workers=num_workers)
        files_locator = process_hdf5_set_files(files_locator, num_workers=num_workers)
        files_locator = process_brainvision_files(files_locator, self.get_shared_attr()['verbose'], num_workers=num_workers)
        files_locator = process_csv_files(
            files_locator,
            num_workers=num_workers,
            min_file_size=min_file_size,
        )
        files_locator = process_wfdb_files(files_locator, num_workers=num_workers)
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
        # --- anchor: EEGParser.get_data kernel hook ---
        row = self.get_shared_attr()['locator'].iloc[data_idx]
        raw = get_data_row(row, **kwargs)
        return apply_dataset_kernel(self.main_instance, raw, row)

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

        def check_channel_counts(row):
            try:
                # Get data_shape and num_channels from the row
                data_shape = row['Data Shape']
                num_channels = row['Number of Channels']

                # Get the channel names from the row and count the number of channels
                channel_names = row['Channel Names']
                num_channels_from_names = len(channel_names.split(','))

                # Convert data_shape into a list of dimensions
                dimensions = [int(dim) for dim in data_shape.strip('()').split(',')]
                # Check if the minimum dimension matches the reported channel counts
                if min(dimensions) == int(float(num_channels)) and min(dimensions) == num_channels_from_names:
                    return []
                else:
                    return ["Mismatch in reported and actual channel counts"]
            except (ValueError, AttributeError, TypeError):
                # Return an error message if there's a value conversion issue
                return ["Channel count or data shape format error"]


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
            errors.extend(check_data_shape(str(row.get('Data Shape', '')).strip()))

            if str(row.get('File Type', '')).strip() == '':
                errors.append("Miss data in File Type")
            elif row['File Type'] == "unidentified":
                errors.append("File Type is 'unidentified'")

            if str(row.get('File Path', '')).strip() == '':
                errors.append("Miss data in File Path")
            elif not os.path.exists(row['File Path']):
                errors.append("File does not exist")

            errors.extend(
                check_channel_counts(row))
            errors.extend(check_duration(str(row.get('Sampling Rate', '')).strip(), str(row.get('Duration', '')).strip(),
                                         str(row.get('Data Shape', '')).strip()))
            errors.extend(check_channel_duplicates(str(row.get('Channel Names', '')).strip()))
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

    montage = create_montage_from_json(os.path.join(current_dir, '..', '..', 'resources', 'combined_montage.json'))
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
    """Apply locator channel schema to one raw object.

    This function keeps EEGUnity's locator-driven design intact:
    ``channel_str`` is treated as the source of truth, and raw metadata is
    overwritten according to the ``type:name`` pairs it contains.

    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw data object to patch.
    channel_str : str
        Comma-separated ``type:name`` string from locator.

    Returns
    -------
    mne.io.Raw
        Patched raw object with renamed channels and updated channel kinds.

    Raises
    ------
    ValueError
        If any channel entry is not in ``type:name`` format or channel count
        mismatches.

    Examples
    --------
    >>> # raw = set_channel_type(raw, "eeg:Fz, stim:event_code")  # doctest: +SKIP
    """
    entries = [entry.strip() for entry in str(channel_str).split(',') if entry.strip()]
    channel_info = [_split_typed_channel(entry) for entry in entries]

    if len(channel_info) != len(raw_data.info['ch_names']):
        raise ValueError(
            "Channel count mismatch between raw data and locator channel string."
        )

    # Rename channels according to locator names (suffix after first ':').
    new_names = [name for _, name in channel_info]
    channel_mapping = {
        raw_name: new_name
        for raw_name, new_name in zip(raw_data.info['ch_names'], new_names)
    }
    raw_data.rename_channels(channel_mapping)

    # Patch channel type fields in-place from locator types.
    for ch_meta, (raw_type, _) in zip(raw_data.info['chs'], channel_info):
        mne_type = _resolve_mne_channel_type(raw_type)
        if mne_type is None:
            warnings.warn(
                f"Unknown channel type prefix '{raw_type}'. Falling back to 'bio'."
            )
            mne_type = 'bio'
        tpl = _channel_type_template(mne_type)
        ch_meta['kind'] = tpl['kind']
        ch_meta['coil_type'] = tpl['coil_type']
        ch_meta['unit'] = tpl['unit']
        ch_meta['unit_mul'] = tpl['unit_mul']
        ch_meta['coord_frame'] = tpl['coord_frame']

    return raw_data


def get_data_row(row: dict,
                 norm_type: str = None,
                 is_set_channel_type: Union[bool, None] = None,
                 is_set_montage: bool = False,
                 pick_types_params: dict = None,
                 unit_convert: str = None,
                 read_raw_params: dict = None,
                 handle_nonstandard_params: dict = None,
                 preload: bool = True) -> mne.io.BaseRaw:
    """
    Process and return raw EEG data based on the input row information.

    This function handles both standard and non-standard data, with options for setting channel types,
    montage, normalization, and unit conversion.

    Parameters
    ----------
    row : dict
        Dictionary containing data attributes, such as file paths, file types, and channel names.
    norm_type : str, optional
        Type of normalization to apply, if any. Defaults to None.
    is_set_channel_type : bool or None, optional
        Determines whether to set channel types based on the provided information.
        - If `True`, channel types will be set explicitly.
        - If `None`, the setting of channel types depends on whether the **File Path** in the locator follows the format
        `"type:name"` (see `UnifiedDataset.EEGBatch.format_channel_names()` for details).
        Defaults to `None`.
    is_set_montage : bool, optional
        Whether to set montage (electrode coordinates). Defaults to False.
    pick_types_params : dict, optional
        Dictionary specifying which channel types to include. The keys should match the parameters of
        `raw.pick_types()`. Defaults to None.
    unit_convert : str, optional
        Conversion type for resetting channel units. Defaults to None.
    read_raw_params : dict, optional
        Additional parameters to pass to `mne.io.read_raw()` for standard data loading.
    handle_nonstandard_params : dict, optional
        Additional parameters to pass to `handle_nonstandard_data()` for non-standard data loading.
    preload : bool, optional
        Whether to preload the data into memory. Defaults to True.

    Returns
    -------
    mne.io.BaseRaw
        The processed raw EEG data object.

    Raises
    ------
    ValueError
        If the number of channels in the locator file does not match the metadata.
    Warning
        If `pick_types` is not None but `is_set_channel_type` is False, a warning will be issued
        to inform the user to set `is_set_channel_type=True`.
    """
    filepath = row.get('File Path')
    file_type = row.get('File Type')

    # Set default parameter dictionaries if None
    if read_raw_params is None:
        read_raw_params = {}
    if handle_nonstandard_params is None:
        handle_nonstandard_params = {}
    read_raw_kwargs = dict(read_raw_params)
    handle_nonstandard_kwargs = dict(handle_nonstandard_params)
    read_raw_verbose = read_raw_kwargs.pop('verbose', 'CRITICAL')
    handle_nonstandard_verbose = handle_nonstandard_kwargs.pop('verbose', 'CRITICAL')

    # Handle standard or non-standard data loading based on file type
    if file_type == "standard_data":  # Load standard data using MNE-Python
        _verbose = read_raw_verbose
        if filepath.endswith('.vhdr'):
            try:
                raw_data = mne.io.read_raw(filepath, verbose=_verbose, preload=preload, **read_raw_kwargs)
            except Exception:
                # BrainVision sidecar path mismatch - retry with patched header in /tmp
                _tmp_path, _extra_tmp = _patch_vhdr(filepath)
                try:
                    raw_data = mne.io.read_raw(_tmp_path, verbose=_verbose, preload=preload, **read_raw_kwargs)
                finally:
                    for _p in [_tmp_path] + _extra_tmp:
                        try:
                            os.remove(_p)
                        except Exception:
                            pass
        elif filepath.endswith('.rec'):
            raw_data = _read_edf_via_tempfile(filepath, verbose=_verbose, preload=preload)
        elif filepath.endswith('.edf'):
            try:
                raw_data = mne.io.read_raw(filepath, verbose=_verbose, preload=preload, **read_raw_kwargs)
            except Exception:
                raw_data = _read_edf_with_patched_header(filepath, verbose=_verbose, preload=preload)
        else:
            raw_data = mne.io.read_raw(filepath, verbose=_verbose, preload=preload, **read_raw_kwargs)
        channel_names = [name.strip() for name in row.get('Channel Names', '').split(',')]
        if len(channel_names) != len(raw_data.info['ch_names']):
            raise ValueError(f"The number of channels in the locator file does not match metadata: {filepath}")
        channel_mapping = {original: new for original, new in zip(raw_data.info['ch_names'], channel_names)}
        raw_data.rename_channels(channel_mapping)
    else:  # Handle non-standard data loading
        raw_data = handle_nonstandard_data(
            row,
            verbose=handle_nonstandard_verbose,
            preload=preload,
            **handle_nonstandard_kwargs,
        )

    # Warn if pick_types is provided but channel type setting is disabled
    if pick_types_params is not None and not is_set_channel_type:
        warnings.warn("When `pick_types` is not None, set `is_set_channel_type=True`.")

    # Set channel types if specified
    is_formated = _is_typed_channel_string(row['Channel Names'])
    if (is_set_channel_type is None and is_formated) or bool(is_set_channel_type):
        raw_data = set_channel_type(raw_data, row['Channel Names'])

    # Apply pick types if provided
    if pick_types_params is not None:
        raw_data = raw_data.pick_types(**pick_types_params)

    # Set montage if specified2
    if is_set_montage:
        raw_data = set_montage_any(raw_data)

    # Apply normalization if specified
    if norm_type and 'MEAN STD' in row:
        raw_data = normalize_data(raw_data, row['MEAN STD'], norm_type)

    # Convert units if required
    if unit_convert and 'Infer Unit' in row:
        raw_data = set_infer_unit(raw_data, row)
        raw_data = convert_unit(raw_data, unit_convert)

    # Correct meas_date if it falls outside valid timestamp range
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
    """Format and standardize channel names into ``type:name`` entries.

    The parser supports two paths:

    1. **Explicit typed input** (recommended): entries already in
       ``type:name`` form, including MNE channel types such as
       ``seeg:LA1``, ``ecog:G1``, ``stim:event_code`` and ``misc:rt``.
       These are preserved (with light prefix canonicalization).
    2. **Heuristic input**: untyped names are classified by EEGUnity rules
       (EEG/EOG/EMG/ECG/STIM/Unknown).

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

    Examples
    --------
    >>> channel_name_parser("Fz, Cz, Pz")
    'eeg:Fz, eeg:Cz, eeg:Pz'
    >>> channel_name_parser("seeg:LA1, seeg:LA2")
    'seeg:LA1, seeg:LA2'
    """
    # Define a function to check if a channel is an EOG channel
    def is_eog_channel(channel):
        return "eog" in channel.lower() or "loc" in channel.lower()  or "roc" in channel.lower()

    # Define a function to check if a channel is an MEG channel
    def is_meg_channel(channel):
        return "meg" in channel.lower()

    # Define a function to check if a channel is an EMG channel
    def is_emg_channel(channel):
        return "emg" in channel.lower()

    # Define a function to check if a channel is an ECG channel
    def is_ecg_channel(channel):
        return "ecg" in channel.lower() or "ekg" in channel.lower() # update: EKG is also known as ECG

    # Define a function to check if a channel is an ECG channel
    def is_stim_channel(channel):
        return "stim" in channel.lower() or "event" in channel.lower() or "marker" in channel.lower()

    # Define a function to check if a channel is an DOUBLE channel
    def is_double_channel(channel):
        if '-' not in channel:
            return False
        parts = channel.split('-')
        if len(parts) != 2:
            return False

        valid_channels = ["REF", "LE", "EEG", "ECG", "EOG", "EMG", "EKG", "MEG"]

        # Convert both parts to upper case for case-insensitive comparison
        part1, part2 = parts[0].upper(), parts[1].upper()

        if part1 in valid_channels or part2 in valid_channels:
            return False
        return True

    # Define a function to standardize channel names
    def standardize_eeg_channel(channel):
        # remove prefix
        channel = channel.replace('EEG:', '').replace('EEG', '').replace('eeg', '')
        return f"eeg:{channel}"

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

        # initialization
        prefixes_suffixes = set(EEG_PREFIXES_SUFFIXES)

        previous_length = -1
        while len(channel) != previous_length:
            previous_length = len(channel)
            for prefix_suffix in prefixes_suffixes:
                if channel.startswith(prefix_suffix):
                    channel = channel[len(prefix_suffix):].strip()
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
        # ------------------------------------------------------------------
        # misc: label channels (MNE type ``misc``) pass through unchanged.
        # They carry continuous float labels added by kernels (e.g. reaction
        # time, inter-event gap) and follow the ``misc:{task_name}`` naming
        # convention.  Unlike STIM channels (integer trigger codes), misc
        # channels are NOT used by mne.find_events / events_from_annotations
        # and are excluded from MNE filter() and ICA by default.
        # See eegunity.utils.label_channel for the full specification.
        # ------------------------------------------------------------------
        if channel.lower().startswith('misc:'):
            formatted_channel = f"misc:{channel[len('misc:'):]}"
            if formatted_channel in seen_channels:
                warnings.warn(f"Duplicate formatted channel detected: {formatted_channel}")
                return input_string
            formatted_channels.append(formatted_channel)
            seen_channels.add(formatted_channel)
            continue

        # ------------------------------------------------------------------
        # Explicit typed channels pass through first.
        # This enables locator values such as "seeg:LA1", "ecog:G1",
        # "fnirs_od:S1_D1 760", etc.
        # ------------------------------------------------------------------
        if ':' in channel:
            try:
                explicit_type, explicit_name = _split_typed_channel(channel)
                resolved_type = _resolve_mne_channel_type(explicit_type)
            except ValueError:
                explicit_type = None
                explicit_name = None
                resolved_type = None

            if resolved_type is not None:
                explicit_prefix = _locator_prefix_from_mne_type(explicit_type, resolved_type)
                formatted_channel = f"{explicit_prefix}:{explicit_name}"
                if formatted_channel in seen_channels:
                    warnings.warn(f"Duplicate formatted channel detected: {formatted_channel}")
                    return input_string
                formatted_channels.append(formatted_channel)
                seen_channels.add(formatted_channel)
                continue
            if explicit_type is not None:
                warnings.warn(
                    f"Unknown channel type prefix '{explicit_type}'. "
                    "Falling back to bio."
                )
                formatted_channel = f"bio:{explicit_name}"
                if formatted_channel in seen_channels:
                    warnings.warn(f"Duplicate formatted channel detected: {formatted_channel}")
                    return input_string
                formatted_channels.append(formatted_channel)
                seen_channels.add(formatted_channel)
                continue

        channel = preprocess_channel(channel)

        if is_eog_channel(channel):
            formatted_channel = f"eog:{channel.replace('EOG:', '').replace('EOG', '').replace('eog', '')}"
        elif is_meg_channel(channel):
            formatted_channel = f"meg:{channel.replace('MEG:', '').replace('MEG', '').replace('meg', '')}"
        elif is_emg_channel(channel):
            formatted_channel = f"emg:{channel.replace('EMG:', '').replace('EMG', '').replace('emg', '')}"
        elif is_ecg_channel(channel):
            formatted_channel = f"ecg:{channel.replace('ECG:', '').replace('ECG', '').replace('ecg', '')}"
        elif is_double_channel(channel):
            formatted_channel = f"eeg:{channel.replace('Dual:', '')}"
        elif is_stim_channel(channel):
            formatted_channel = f"stim:{channel.replace('STIM:', '')}"
        else:
            re_channel = remap_standard_name(channel)
            matched = False
            for standard_name in STANDARD_EEG_CHANNELS:
                if re_channel.lower() == standard_name.lower():
                    formatted_channel = standardize_eeg_channel(standard_name)
                    matched = True
                    break
            if not matched:
                formatted_channel = f"bio:{channel.replace('Unknown:', '').replace('bio:', '').replace('BIO:', '')}"
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


def _read_edf_via_tempfile(filepath, verbose='CRITICAL', preload=False):
    """Read EDF content saved with a non-standard file extension.

    Parameters
    ----------
    filepath : str
        Source path, commonly a ``.rec`` file that actually stores EDF bytes.
    verbose : str, optional
        MNE verbosity level.
    preload : bool, optional
        Whether to preload data into memory.

    Returns
    -------
    mne.io.BaseRaw
        Parsed EDF raw object.

    Examples
    --------
    >>> raw = _read_edf_via_tempfile("sample.rec")  # doctest: +SKIP
    """
    import tempfile

    with open(filepath, 'rb') as fh:
        raw_bytes = fh.read()

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        return mne.io.read_raw_edf(tmp_path, preload=preload, verbose=verbose)
    finally:
        os.remove(tmp_path)


def _read_edf_with_patched_header(filepath, verbose='CRITICAL', preload=False):
    """Read EDF files that use ``:`` instead of ``.`` in date/time header fields.

    Parameters
    ----------
    filepath : str
        EDF file path.
    verbose : str, optional
        MNE verbosity level.
    preload : bool, optional
        Whether to preload data into memory.

    Returns
    -------
    mne.io.BaseRaw
        Parsed EDF raw object.

    Examples
    --------
    >>> raw = _read_edf_with_patched_header("sample.edf")  # doctest: +SKIP
    """
    import tempfile

    with open(filepath, 'rb') as fh:
        raw_bytes = bytearray(fh.read())

    for start in (168, 176):
        field = bytes(raw_bytes[start:start + 8])
        if b':' in field:
            raw_bytes[start:start + 8] = field.replace(b':', b'.')

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
        tmp.write(bytes(raw_bytes))
        tmp_path = tmp.name

    try:
        return mne.io.read_raw_edf(tmp_path, preload=preload, verbose=verbose)
    finally:
        os.remove(tmp_path)


def handle_nonstandard_data(row, verbose='CRITICAL', preload=True):
    """Load non-standard EEG files and return an ``mne.io.Raw`` object.

    Supported paths include MATLAB ``.mat``, HDF5 EEGLAB ``.set`` rows marked
    as ``eeglab_hdf5``, CSV/TXT rows marked as ``csvData``, WFDB ``.hea`` rows
    marked as ``wfdbData``, and EDF content saved as ``.rec``.

    Parameters
    ----------
    row : pandas.Series
        Locator row containing at least ``File Path``, ``Channel Names``,
        ``Sampling Rate``, and ``File Type``.
    verbose : str, optional
        MNE verbosity level. Defaults to ``'CRITICAL'``.
    preload : bool, optional
        Whether to preload HDF5 EEGLAB data when reading ``eeglab_hdf5`` rows.

    Returns
    -------
    mne.io.BaseRaw
        Parsed raw object.

    Examples
    --------
    >>> raw = handle_nonstandard_data(locator_row, preload=False)  # doctest: +SKIP
    """
    filepath = row['File Path']
    file_type = str(row.get('File Type', ''))

    if file_type == 'eeglab_hdf5':
        return read_eeglab_hdf5(filepath, preload=preload, verbose=verbose)

    if filepath.endswith('.rec'):
        return _read_edf_via_tempfile(filepath, verbose=verbose, preload=preload)

    if filepath.endswith('.edf'):
        return _read_edf_with_patched_header(filepath, verbose=verbose, preload=preload)

    if filepath.endswith('.mat'):
        matdata = scipy.io.loadmat(filepath)
        eeg_data = _find_variables_by_condition(
            matdata,
            _condition_source_data,
            max_depth=5,
            max_width=20,
        )[1]
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        channel_names = row['Channel Names'].split(',')
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=float(row['Sampling Rate']),
            ch_types='eeg',
            verbose=verbose,
        )
        return mne.io.RawArray(eeg_data, info)

    if (filepath.endswith('.csv') or filepath.endswith('.txt')) and file_type == 'csvData':
        header_option = None if row.get('Header', 'infer') == 'None' else 'infer'
        df = pd.read_csv(filepath, header=header_option, skipinitialspace=True)

        if header_option is None:
            df.columns = [str(i) for i in range(1, len(df.columns) + 1)]

        channel_names = [name.strip() for name in row['Channel Names'].split(',')]
        sfreq = float(row['Sampling Rate'])
        if not all(name in df.columns for name in channel_names):
            raise ValueError(
                f"Number of channels marked in locator does not match metadata: {filepath}"
            )

        eeg_data = df[channel_names].values.T
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        return mne.io.RawArray(eeg_data, info)

    if filepath.endswith('.hea') and file_type == 'wfdbData':
        import wfdb

        record_name = os.path.splitext(filepath)[0]
        record = wfdb.rdrecord(record_name)
        eeg_data = record.p_signal.T
        channel_names = [name.strip() for name in row['Channel Names'].split(',')]
        sfreq = float(row['Sampling Rate'])
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg', verbose=verbose)
        return mne.io.RawArray(eeg_data, info)

    raise ValueError("Parsing of files other than .mat/.csv/.txt/.set/.hea/.edf/.rec is not supported")


def extract_events(raw, event_source: str = 'auto', stim_channel=None, regexp: str = r'^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$'):
    """Extract events from annotations and/or stim channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object.
    event_source : {'auto', 'annotations', 'stim'}, optional
        Event source strategy:

        - ``'annotations'``: use :func:`mne.events_from_annotations` only.
        - ``'stim'``: use :func:`mne.find_events` only.
        - ``'auto'`` (default): annotations first, then stim fallback.
    stim_channel : str | list[str] | None, optional
        Stim channel name(s) passed to :func:`mne.find_events`. If ``None``,
        all channels with MNE type ``stim`` are used automatically.
    regexp : str, optional
        Regular expression passed to :func:`mne.events_from_annotations`.
        Defaults to MNE's standard pattern that skips ``bad``/``edge`` labels.

    Returns
    -------
    events : numpy.ndarray
        Events array shaped ``(n_events, 3)``.
    event_id : dict
        Event-id mapping dictionary.

    Raises
    ------
    ValueError
        If ``event_source`` is not one of the accepted values.

    Examples
    --------
    >>> events, event_id = extract_events(raw, event_source='auto')  # doctest: +SKIP
    >>> events, event_id = extract_events(raw, event_source='stim', stim_channel=['TRIG'])  # doctest: +SKIP
    """
    event_source = str(event_source).strip().lower()
    if event_source not in {'auto', 'annotations', 'stim'}:
        raise ValueError("event_source must be one of 'auto', 'annotations', or 'stim'.")

    def _from_annotations():
        try:
            events_, event_id_ = mne.events_from_annotations(raw, regexp=regexp)
            if events_.size == 0 or not event_id_:
                return None, None
            return events_, event_id_
        except Exception:
            return None, None

    def _from_stim():
        try:
            if stim_channel is not None:
                stim_ch = stim_channel
            else:
                stim_ch = [
                    raw.ch_names[i]
                    for i in range(len(raw.ch_names))
                    if mne.channel_type(raw.info, i) == 'stim'
                ]
                if not stim_ch:
                    stim_ch = None
            events_ = mne.find_events(raw, stim_channel=stim_ch)
            if events_.size == 0:
                return None, None
            unique_ids = np.unique(events_[:, 2])
            event_id_ = {f'event_{eid}': int(eid) for eid in unique_ids}
            return events_, event_id_
        except Exception:
            return None, None

    if event_source == 'annotations':
        events, event_id = _from_annotations()
    elif event_source == 'stim':
        events, event_id = _from_stim()
    else:  # auto
        events, event_id = _from_annotations()
        if events is None:
            events, event_id = _from_stim()

    if events is None or event_id is None:
        print("Events Not Found")
        return np.empty((0, 3), dtype=int), {}
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

def _patch_vhdr(vhdr_path):
    """Create a patched copy of a BrainVision .vhdr file with corrected sidecar paths.

    BIDS tools sometimes rename .vhdr/.eeg/.vmrk files but do not update the
    internal DataFile= and MarkerFile= references inside the .vhdr header.
    This function writes corrected temporary files to the system temp directory
    using **absolute paths** for DataFile and MarkerFile so that MNE can
    locate the companion files regardless of the working directory.

    If the companion .vmrk file does not exist a minimal valid empty .vmrk is
    created so that MNE's mandatory MarkerFile requirement is satisfied.

    Parameters
    ----------
    vhdr_path : str
        Absolute path to the original .vhdr file.

    Returns
    -------
    (str, list) or (None, [])
        ``(tmp_vhdr_path, extra_tmp_paths)`` where ``extra_tmp_paths`` is a
        list of any additional temp files created (e.g. dummy .vmrk).
        Returns ``(None, [])`` if the file could not be read.
        Caller is responsible for deleting all returned paths.

    Examples
    --------
    >>> _patch_vhdr("sample.vhdr")  # doctest: +SKIP
    """
    import tempfile

    stem = os.path.splitext(vhdr_path)[0]
    # Use absolute paths - MNE resolves DataFile as
    # os.path.join(vhdr_dir, DataFile); on POSIX,
    # os.path.join('/tmp/', '/abs/path') == '/abs/path'.
    abs_eeg = stem + '.eeg'
    abs_vmrk = stem + '.vmrk'
    extra_tmp = []

    try:
        with open(vhdr_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return None, []

    # If the .vmrk file doesn't exist, create a minimal valid empty .vmrk
    if not os.path.isfile(abs_vmrk):
        try:
            vmrk_tmp = tempfile.NamedTemporaryFile(
                suffix='.vmrk',
                dir=tempfile.gettempdir(),
                delete=False,
                mode='w',
                encoding='utf-8',
            )
            vmrk_tmp.write(
                "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
                "[Common Infos]\nCodepage=UTF-8\n"
                f"DataFile={abs_eeg}\n\n"
                "[Marker Infos]\n; No markers\n"
            )
            vmrk_tmp.close()
            abs_vmrk = vmrk_tmp.name
            extra_tmp.append(vmrk_tmp.name)
        except Exception:
            pass  # Proceed without marker file; MNE may still fail gracefully

    # Patch DataFile reference to absolute path
    patched = re.sub(r'^(DataFile\s*=).*$',
                     r'\g<1>' + abs_eeg,
                     content, flags=re.MULTILINE)

    # Patch or insert MarkerFile reference to absolute path
    if re.search(r'^MarkerFile\s*=', patched, flags=re.MULTILINE):
        patched = re.sub(r'^(MarkerFile\s*=).*$',
                         r'\g<1>' + abs_vmrk,
                         patched, flags=re.MULTILINE)
    else:
        # Insert after DataFile line
        patched = re.sub(r'^(DataFile\s*=.*$)',
                         r'\1\nMarkerFile=' + abs_vmrk,
                         patched, flags=re.MULTILINE)

    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix='.vhdr',
            dir=tempfile.gettempdir(),
            delete=False,
            mode='w',
            encoding='utf-8',
        )
        tmp.write(patched)
        tmp.close()
        return tmp.name, extra_tmp
    except Exception:
        return None, extra_tmp


def _process_single_brainvision_file(vhdr_path, verbose):
    """Retry reading a failed .vhdr file after patching sidecar references.

    Parameters
    ----------
    vhdr_path : str
        Path to the .vhdr file.
    verbose : str
        MNE verbosity level.

    Returns
    -------
    dict
        Metadata dict (same schema as _process_single_mne_file) or a dict
        with File Type='unknown' and an Error entry on failure.

    Examples
    --------
    >>> _process_single_brainvision_file("sample.vhdr", "CRITICAL")  # doctest: +SKIP
    """
    tmp_path, extra_tmp = _patch_vhdr(vhdr_path)
    if tmp_path is None:
        return {'File Type': 'unknown', 'Error': 'vhdr patch failed'}

    try:
        data = mne.io.read_raw(tmp_path, verbose=verbose)
        nchan = int(data.info['nchan'])
        n_times = int(data.n_times)
        result = {
            'File Type': 'standard_data',
            'Data Shape': str((nchan, n_times)),
            'Channel Names': ', '.join(data.info.get('ch_names', [])),
            'Number of Channels': nchan,
            'Sampling Rate': float(data.info.get('sfreq', 0.0)),
            'Duration': float(data.times[-1]) if len(data.times) > 0 else 0.0,
        }
        return result
    except Exception as e:
        return {'File Type': 'unknown', 'Error': f'vhdr patched read failed: {e}'}
    finally:
        for p in [tmp_path] + extra_tmp:
            try:
                os.remove(p)
            except Exception:
                pass


def process_brainvision_files(files_locator, verbose, num_workers=0):
    """Retry failed BrainVision .vhdr files by patching internal sidecar paths.

    Targets .vhdr files that failed MNE reading because internal DataFile= or
    MarkerFile= paths reference pre-BIDS filenames that no longer exist.

    Parameters
    ----------
    files_locator : pandas.DataFrame
        Locator DataFrame; must already contain an 'Error' column.
    verbose : str
        MNE verbosity level.
    num_workers : int, optional
        Number of parallel worker threads (0 = sequential).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with metadata filled for successfully re-read files.

    Examples
    --------
    >>> process_brainvision_files(locator_df, "CRITICAL", num_workers=2)  # doctest: +SKIP
    """
    if 'Error' not in files_locator.columns:
        return files_locator

    eligible = []
    for idx, row in files_locator.iterrows():
        path = row['File Path']
        error = str(row.get('Error', ''))
        if (path.endswith('.vhdr')
                and str(row.get('File Type', 'unknown')) == 'unknown'
                and ('No such file or directory' in error
                     or 'markerfile' in error.lower())):
            eligible.append((idx, path))

    if not eligible:
        return files_locator

    indices, file_paths = zip(*eligible)

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                lambda fp: _process_single_brainvision_file(fp, verbose),
                file_paths,
            ))
    else:
        results = [_process_single_brainvision_file(fp, verbose) for fp in file_paths]

    for idx, result in zip(indices, results):
        for key, value in result.items():
            files_locator.at[idx, key] = pd.NA if pd.isna(value) else str(value)

    return files_locator


def _process_single_mne_file(filepath, verbose):
    """
    Process a single file with MNE and return metadata dict.

    Parameters
    ----------
    filepath : str
        Path to the file to process.
    verbose : str
        Verbosity level for MNE functions.

    Returns
    -------
    dict
        A dictionary containing extracted metadata or error information.

    Examples
    --------
    >>> _process_single_mne_file("sample.edf", "CRITICAL")  # doctest: +SKIP
    """
    def _make_result(data):
        nchan = int(data.info['nchan'])
        return {
            'File Type': 'standard_data',
            'Data Shape': str((nchan, int(data.n_times))),
            'Channel Names': ', '.join(data.info.get('ch_names', [])),
            'Number of Channels': nchan,
            'Sampling Rate': float(data.info.get('sfreq', 0.0)),
            'Duration': float(data.times[-1]) if len(data.times) > 0 else 0.0,
        }

    try:
        data = mne.io.read_raw(filepath, verbose=verbose)
        result = _make_result(data)
        print(f"Retrieved channel sequence: {data.info.get('ch_names', [])}")
        return result
    except Exception as first_exc:
        if filepath.endswith('.rec'):
            try:
                data = _read_edf_via_tempfile(filepath, verbose=verbose)
                result = _make_result(data)
                print(f"Retrieved channel sequence (via EDF fallback): {data.info.get('ch_names', [])}")
                return result
            except Exception:
                pass

        if filepath.endswith('.edf'):
            try:
                data = _read_edf_with_patched_header(filepath, verbose=verbose)
                result = _make_result(data)
                print(f"Retrieved channel sequence (via EDF header patch): {data.info.get('ch_names', [])}")
                return result
            except Exception:
                pass

        # These extensions have dedicated downstream processors and always fail
        # MNE's generic reader — suppress the noisy print to avoid misleading output.
        _HANDLED_ELSEWHERE = ('.mat', '.hea', '.csv', '.txt')
        if not filepath.endswith(_HANDLED_ELSEWHERE):
            print(f"Failed to process file {filepath}: {first_exc}")
        return {
            'File Type': 'unknown',
            'Data Shape': 'error',
            'Error': str(first_exc),
        }


def process_mne_files(files_locator, verbose, num_workers=0):
    """
    Process MNE files based on a locator DataFrame.

    Parameters
    ----------
    files_locator : pandas.DataFrame
        DataFrame containing file paths and related metadata for processing.
    verbose : str
        Verbosity level for MNE functions.
    num_workers : int, optional
        Number of worker threads for parallel processing (default is 0, sequential).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with metadata extracted from processed files.

    Examples
    --------
    >>> process_mne_files(locator_df, verbose="CRITICAL", num_workers=0)  # doctest: +SKIP
    """
    indices = list(files_locator.index)
    filepaths = [files_locator.at[idx, 'File Path'] for idx in indices]

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(lambda fp: _process_single_mne_file(fp, verbose), filepaths))
    else:
        results = [_process_single_mne_file(fp, verbose) for fp in filepaths]

    for idx, result in zip(indices, results):
        for key, value in result.items():
            files_locator.at[idx, key] = pd.NA if pd.isna(value) else str(value)

    return files_locator
