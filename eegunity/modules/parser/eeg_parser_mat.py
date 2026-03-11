import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numpy import ndarray
from scipy.io import loadmat


def _is_numeric(s):
    """Match integer or floating-point numbers."""
    pattern = r'^-?\d+(\.\d+)?$'
    return bool(re.match(pattern, s))


def _load_mat_h5py(file_path):
    """Read a MATLAB v7.3 (HDF5) ``.mat`` file using ``h5py``.

    Parameters
    ----------
    file_path : str
        Path to the MATLAB v7.3 HDF5 file.

    Returns
    -------
    dict or None
        Parsed nested dictionary on success, otherwise ``None``.

    Examples
    --------
    >>> _load_mat_h5py("example_v73.mat")  # doctest: +SKIP
    """
    try:
        import h5py
    except ImportError:
        return None

    def _visit(item):
        if isinstance(item, h5py.Dataset):
            return item[()]
        if isinstance(item, h5py.Group):
            return {k: _visit(v) for k, v in item.items()}
        return item

    skip_keys = {"__header__", "__version__", "__globals__"}
    try:
        with h5py.File(file_path, "r") as fh:
            return {k: _visit(v) for k, v in fh.items() if k not in skip_keys}
    except Exception:
        return None


def _load_mat_file(file_path):
    """Load a MATLAB ``.mat`` file with automatic format detection.

    Parameters
    ----------
    file_path : str
        Path to a ``.mat`` file.

    Returns
    -------
    dict or None
        Parsed data dictionary on success, otherwise ``None``.

    Examples
    --------
    >>> _load_mat_file("example.mat")  # doctest: +SKIP
    """
    try:
        return loadmat(file_path, simplify_cells=True)
    except NotImplementedError:
        return _load_mat_h5py(file_path)
    except Exception:
        return None


def _process_single_mat_file(file_path):
    """
    Process a single MAT file and return metadata dict or None.

    Parameters
    ----------
    file_path : str
        Path to the MAT file.

    Returns
    -------
    dict or None
        A dictionary containing extracted metadata, or None if the file cannot be processed.
    """
    file_size = os.path.getsize(file_path)
    if file_size <= 5 * 1024 * 1024:
        return None

    data = _load_mat_file(file_path)
    if data is None:
        print(f"    [eeg_parser_mat] Skipping {file_path}: cannot parse MAT file.")
        return None
    channel_name = _find_variables_by_condition(data, _condition_sampling_channel_name,
                                                max_depth=5, max_width=20)
    sampling_rate = _find_variables_by_condition(data, _condition_sampling_rate,
                                                 max_depth=5, max_width=20)
    source_data = _find_variables_by_condition(data, _condition_source_data,
                                               max_depth=5, max_width=20)
    source_data_3d = _find_variables_by_condition(data, _condition_source_data_3d,
                                                  max_depth=5, max_width=20)

    result = {}
    if isinstance(source_data[1], ndarray):
        result['Sampling Rate'] = str(sampling_rate[1]).strip("HhZz")
        if isinstance(channel_name[1], ndarray):
            result['Channel Names'] = ','.join(str(x) for x in channel_name[1])
        result['Number of Channels'] = str(min(source_data[1].shape))
        result['Data Shape'] = str(source_data[1].shape)
        if _is_numeric(result['Sampling Rate']):
            result['Duration'] = str(max(source_data[1].shape) / float(result['Sampling Rate']))
        else:
            result['Duration'] = ''
        result['File Type'] = "matRawData:" + str(source_data[0])
        return result
    elif isinstance(source_data_3d[1], ndarray):
        result['Sampling Rate'] = str(sampling_rate[1]).strip("HhZz")
        if isinstance(channel_name[1], ndarray):
            print(','.join(str(x) for x in channel_name[1]))
            result['Channel Names'] = ','.join(str(x) for x in channel_name[1])
        result['Number of Channels'] = str(len(channel_name[1]))
        result['Data Shape'] = str(source_data_3d[1].shape)
        if _is_numeric(result['Sampling Rate']):
            result['Duration'] = str(max(source_data_3d[1].shape) / float(result['Sampling Rate']))
        else:
            result['Duration'] = ''
        result['File Type'] = "matEpochData:" + str(source_data_3d[0])
        return result
    else:
        return None


def process_hdf5_set_files(files_locator, num_workers=0):
    """Process EEGLAB .set files saved in HDF5 (MATLAB v7.3) format.

    Targets .set files that failed MNE reading with a "HDF reader" error.
    Extracts metadata (channels, srate, duration) via h5py without loading
    raw signal data.

    Parameters
    ----------
    files_locator : pandas.DataFrame
        Locator DataFrame; must already contain an 'Error' column populated by
        process_mne_files().
    num_workers : int, optional
        Number of parallel worker threads (0 = sequential).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with metadata filled for readable HDF5 .set files.

    Examples
    --------
    >>> process_hdf5_set_files(locator_df, num_workers=2)  # doctest: +SKIP
    """
    if 'Error' not in files_locator.columns:
        return files_locator

    eligible = []
    for idx, row in files_locator.iterrows():
        path = row['File Path']
        error = str(row.get('Error', ''))
        if (path.endswith('.set')
                and str(row.get('File Type', 'unknown')) == 'unknown'
                and 'HDF reader' in error):
            eligible.append((idx, path))

    if not eligible:
        return files_locator

    indices, file_paths = zip(*eligible)

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_single_hdf5_set_file, file_paths))
    else:
        results = [_process_single_hdf5_set_file(fp) for fp in file_paths]

    for idx, result in zip(indices, results):
        if result is not None:
            for key, value in result.items():
                files_locator.at[idx, key] = '' if value is None else str(value)

    return files_locator


def _process_single_hdf5_set_file(file_path):
    """Extract metadata from a single HDF5-format EEGLAB .set file.

    Parameters
    ----------
    file_path : str
        Path to the .set file.

    Returns
    -------
    dict or None
        Metadata dict with keys compatible with the locator DataFrame, or None
        if the file cannot be parsed.

    Examples
    --------
    >>> _process_single_hdf5_set_file("sample.set")  # doctest: +SKIP
    """
    try:
        import h5py
    except ImportError:
        print("    [eeg_parser_mat] h5py not installed; cannot read HDF5 .set files.")
        return None

    try:
        with h5py.File(file_path, 'r') as hf:
            srate = float(np.squeeze(hf['srate'][()]))
            nbchan = int(np.squeeze(hf['nbchan'][()]))
            pnts = int(np.squeeze(hf['pnts'][()]))

            # Channel labels: stored as (nchan, 1) object-reference array
            labels = []
            try:
                lab = hf['chanlocs']['labels']  # shape (nchan, 1)
                for i in range(lab.shape[0]):
                    ref = lab[i, 0]
                    chars = hf[ref][:]
                    label = ''.join(chr(int(c)) for c in chars.flatten())
                    labels.append(label.strip())
            except Exception:
                pass

            if len(labels) != nbchan:
                labels = [f'Ch{i + 1}' for i in range(nbchan)]

            return {
                'File Type': 'eeglab_hdf5',
                'Sampling Rate': str(srate),
                'Number of Channels': str(nbchan),
                'Channel Names': ','.join(labels),
                'Data Shape': f'({nbchan}, {pnts})',
                'Duration': str(pnts / srate),
            }
    except Exception as e:
        print(f"    [eeg_parser_mat] HDF5 .set skip {file_path}: {e}")
        return None


def read_eeglab_hdf5(filepath, preload=True, verbose='CRITICAL'):
    """Read a HDF5-format EEGLAB .set file into an MNE RawArray.

    Used by handle_nonstandard_data() when file_type == 'eeglab_hdf5'.
    When preload=False a zero-filled array is returned (metadata + annotations
    only), which is sufficient for kernels that only need sidecar files and
    raw.annotations.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 .set file.
    preload : bool, optional
        If True, load the full EEG signal. If False, return a stub RawArray
        with annotations only (faster, suitable for metadata-only kernels).
    verbose : str, optional
        MNE verbosity level.

    Returns
    -------
    mne.io.RawArray
        MNE Raw object with channel info and (when available) annotations.

    Examples
    --------
    >>> raw = read_eeglab_hdf5("sample.set", preload=False)  # doctest: +SKIP
    """
    import h5py
    import mne

    with h5py.File(filepath, 'r') as hf:
        srate = float(np.squeeze(hf['srate'][()]))
        nbchan = int(np.squeeze(hf['nbchan'][()]))
        pnts = int(np.squeeze(hf['pnts'][()]))

        # Channel labels: stored as (nchan, 1) object-reference array
        labels = []
        try:
            lab = hf['chanlocs']['labels']  # shape (nchan, 1)
            for i in range(lab.shape[0]):
                ref = lab[i, 0]
                chars = hf[ref][:]
                label = ''.join(chr(int(c)) for c in chars.flatten())
                labels.append(label.strip())
        except Exception:
            pass
        if len(labels) != nbchan:
            labels = [f'Ch{i + 1}' for i in range(nbchan)]

        info = mne.create_info(ch_names=labels, sfreq=srate,
                               ch_types='eeg', verbose=verbose)

        if preload:
            # Data shape in HDF5 is (pnts, nchan), transpose to (nchan, pnts).
            data = np.array(hf['data'], dtype=np.float64).T
        else:
            data = np.zeros((nbchan, 1), dtype=np.float64)

        raw = mne.io.RawArray(data, info, verbose=verbose)

        # Extract event annotations from the EEGLAB event structure
        try:
            onsets, durations, descriptions = [], [], []
            evt = hf.get('event')
            if evt is not None and 'latency' in evt and 'type' in evt:
                lats = evt['latency']
                typs = evt['type']
                for i in range(len(lats)):
                    try:
                        if lats.dtype.kind == 'O':
                            lat_val = float(np.squeeze(hf[lats[i]][()]))
                        else:
                            lat_val = float(lats[i])
                        onset = lat_val / srate

                        if typs.dtype.kind == 'O':
                            typ_chars = hf[typs[i]][:]
                            desc = ''.join(chr(int(c)) for c in typ_chars.flatten())
                        else:
                            desc = str(typs[i])

                        onsets.append(onset)
                        durations.append(0.0)
                        descriptions.append(desc.strip())
                    except Exception:
                        continue

            if onsets:
                raw.set_annotations(
                    mne.Annotations(onsets, durations, descriptions))
        except Exception:
            pass

    return raw


def process_mat_files(files_locator, num_workers=0):
    """
    Process MAT files and update a DataFrame with file details.

    Parameters
    ----------
    files_locator : pandas.DataFrame
        A DataFrame containing the metadata of files, including their file paths and other details.
        The column 'File Path' is expected to contain paths to the MAT files.
    num_workers : int, optional
        Number of worker threads for parallel processing (default is 0, sequential).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with additional columns 'File Type', 'Sampling Rate', 'Channel Names', 'Number of Channels', and 'Duration' for each file.
        If a file cannot be processed, appropriate messages are printed.

    Raises
    ------
    FileNotFoundError
        If the MAT file cannot be located.
    Exception
        General exception for unexpected errors during file processing.

    Examples
    --------
    >>> process_mat_files(locator_df, num_workers=0)  # doctest: +SKIP
    """
    # Collect indices of eligible files
    eligible = []
    for index, row in files_locator.iterrows():
        file_path = row['File Path']
        file_type = row['File Type']
        if file_path.endswith('.mat') and file_type == 'unknown':
            eligible.append((index, file_path))

    if not eligible:
        return files_locator

    indices, file_paths = zip(*eligible)

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_single_mat_file, file_paths))
    else:
        results = [_process_single_mat_file(fp) for fp in file_paths]

    for idx, result in zip(indices, results):
        if result is not None:
            for key, value in result.items():
                files_locator.at[idx, key] = '' if value is None else str(value)

    return files_locator


def _find_variables_by_condition(data, condition_func, max_depth=5, max_width=5, debug=False):
    """
    Search for variables in a nested data structure that satisfy a given condition.

    Parameters
    ----------
    data : dict
        The data structure to search through, typically loaded from a .mat file.
    condition_func : function
        A function that takes in a variable's path and value, and returns a boolean indicating whether the variable meets the specified condition.
    max_depth : int, optional
        The maximum depth to search within the nested structure. Defaults to 5.
    max_width : int, optional
        The maximum number of items to check at each depth level. Defaults to 5.
    debug : bool, optional
        If True, enables additional logging for debugging purposes. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the first variable's name and its value that satisfies the condition.
        If no variable satisfies the condition, returns ("unknown", '').
    """
    satisfying_variables = []
    _search_data(data, '', condition_func, satisfying_variables, 0, max_depth, max_width, debug=debug)
    if len(satisfying_variables) == 0:
        return "unknown", ''
    else:
        return satisfying_variables[0]


def _condition_source_data(var_path, var_value):
    """
    Condition function to check if a variable is a 2D ndarray of significant size.

    Parameters
    ----------
    var_path : str
        The path of the variable within the data structure.
    var_value : any
        The value of the variable, typically an array or other data structure.

    Returns
    -------
    bool
        True if `var_value` is a 2D ndarray larger than 5MB, otherwise False.
    """
    if isinstance(var_value, ndarray) and var_value.ndim == 2 and var_value.nbytes > 5 * 1024 * 1024:
        return True
    return False


def _condition_source_data_3d(var_path, var_value):
    """
    Condition function to check if a variable is a 3D ndarray of significant size.

    Parameters
    ----------
    var_path : str
        The path of the variable within the data structure.
    var_value : any
        The value of the variable, typically an array or other data structure.

    Returns
    -------
    bool
        True if `var_value` is a 3D ndarray larger than 5MB, otherwise False.
    """
    if isinstance(var_value, ndarray) and var_value.ndim == 3 and var_value.nbytes > 5 * 1024 * 1024:
        return True
    return False


def _condition_sampling_rate(var_path, var_value):
    """
    Condition function that checks if the variable path contains
    sampling-rate related keywords.

    Parameters
    ----------
    var_path : str
        The path of the variable.
    var_value : any
        The value of the variable (unused in this condition).

    Returns
    -------
    bool
        True if ``'fs'``, ``'fre'``, or ``'rate'`` is in the variable path.
    """
    var_path = var_path.lower()
    return 'fs' in var_path or 'fre' in var_path or 'rate' in var_path


def _condition_sampling_channel_name(var_path, var_value):
    """
    Condition function that checks if the variable path contains 'chan'.

    Parameters
    ----------
    var_path : str
        The path of the variable.
    var_value: any
        The value of the variable (unused in this condition).

    Returns
    -------
    bool
        True if 'chan' is in the variable path, False otherwise.
    """
    var_path = var_path.lower()
    return ('chan' in var_path or 'chname' in var_path or 'clab' in var_path) and (
        isinstance(var_value, ndarray)) and var_value.shape[0] > 2


def _search_data(data, path, condition_func, satisfying_variables, current_depth=0, max_depth=5, max_width=5,
                 ignore_keys=None, debug=False):
    """
    Recursively search for variables in nested data structures that satisfy a given condition.

    Parameters
    ----------
    data : dict or ndarray
        The data structure to search through, which can be a dictionary or a NumPy ndarray.
    path : str
        The current path in the data structure, used for tracking the location of found variables.
    condition_func : callable
        A function that takes a path and data item and returns True if the item satisfies the search condition.
    satisfying_variables : list
        A list that will be populated with tuples of paths and their corresponding data that satisfy the condition.
    current_depth : int, optional
        The current depth of recursion, default is 0.
    max_depth : int, optional
        The maximum depth to search, default is 5.
    max_width : int, optional
        The maximum number of items to process at each level, default is 5.
    ignore_keys : list, optional
        A list of keys to ignore during the search, default excludes certain internal keys.
    debug : bool, optional
        If True, prints debugging information during the search.

    Returns
    -------
    None
        The function modifies the satisfying_variables list in place.
    """
    if ignore_keys is None:
        ignore_keys = ['__header__', '__version__', '__globals__']
    if debug:
        print(f"Searching path: {path}, Current depth: {current_depth}")
    if current_depth >= max_depth:
        if debug:
            print(f"Reached maximum depth, stopping search. Current path: {path}")
        return  # Stop search if maximum depth is reached

    if isinstance(data, dict):
        for key, value in list(data.items())[:max_width]:
            if key in ignore_keys:
                continue
            new_path = f"{path}.{key}" if path else key
            _search_data(value, new_path, condition_func, satisfying_variables, current_depth + 1, max_depth,
                         max_width, ignore_keys, debug)
    elif isinstance(data, list):
        if current_depth < max_depth:
            for i, item in enumerate(data[:max_width]):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                _search_data(item, new_path, condition_func, satisfying_variables, current_depth + 1,
                             max_depth, max_width, ignore_keys, debug)
    elif isinstance(data, ndarray):
        if data.dtype.names is not None:  # Structured array
            for name in list(data.dtype.names)[:max_width]:
                try:
                    nested_value = data[name][0] if data[name].size == 1 else data[name]
                except (AttributeError, IndexError):
                    nested_value = data[name]  # Handle potential access to non-ndarray types
                new_path = f"{path}.{name}" if path else name
                _search_data(nested_value, new_path, condition_func, satisfying_variables, current_depth + 1,
                             max_depth, max_width, ignore_keys, debug)
        else:  # Regular ndarray
            if current_depth < max_depth:
                for i, item in enumerate(data[:max_width]):
                    try:
                        item = item.item() if np.isscalar(item) else item
                    except AttributeError:
                        pass  # item may not be an ndarray
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    _search_data(item, new_path, condition_func, satisfying_variables, current_depth + 1,
                                 max_depth, max_width, ignore_keys, debug)
    if condition_func(path, data):
        if debug:
            print(f"Found a variable satisfying the condition: {path}")  # Print the path of the satisfying variable
        satisfying_variables.append((path, data))
    elif not isinstance(data, (ndarray, dict)) and debug:
        print(f"Non-target type (not ndarray or dict), stopping search. Current path: {path}")

