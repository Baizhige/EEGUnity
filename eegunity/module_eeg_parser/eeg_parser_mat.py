import os
import re

import numpy as np
from numpy import ndarray
from scipy.io import loadmat


def process_mat_files(files_locator):
    def is_numeric(s):
        # Match integer or floating-point numbers
        pattern = r'^-?\d+(\.\d+)?$'
        return bool(re.match(pattern, s))

    for index, row in files_locator.iterrows():
        file_path = row['File Path']
        file_type = row['File Type']

        if file_path.endswith('.mat') and file_type == 'unknown':
            file_size = os.path.getsize(file_path)
            if file_size > 5 * 1024 * 1024:  # Greater than 5MB
                data = loadmat(file_path, simplify_cells=True)  # Simplify dictionary structure
                channel_name = _find_variables_by_condition(data, _condition_sampling_channel_name,
                                                            max_depth=5, max_width=20)
                sampling_rate = _find_variables_by_condition(data, _condition_sampling_rate,
                                                             max_depth=5, max_width=20)
                source_data = _find_variables_by_condition(data, _condition_source_data,
                                                           max_depth=5, max_width=20)
                source_data_3d = _find_variables_by_condition(data, _condition_source_data_3d,
                                                              max_depth=5, max_width=20)
                if isinstance(source_data[1], ndarray):
                    files_locator.at[index, 'Sampling Rate'] = str(sampling_rate[1]).strip("HhZz")
                    if isinstance(channel_name[1], ndarray):
                        # If channel_name is ndarray, ensure correct processing and conversion to string
                        # print(','.join(str(x) for x in channel_name[1]))
                        files_locator.at[index, 'Channel Names'] = ','.join(str(x) for x in channel_name[1])

                    files_locator.at[index, 'Number of Channels'] = str(min(source_data[1].shape))
                    files_locator.at[index, 'Data Shape'] = str(source_data[1].shape)
                    if is_numeric(files_locator.at[index, 'Sampling Rate']):
                        files_locator.at[index, 'Duration'] = str(
                            max(source_data[1].shape) / float(files_locator.at[index, 'Sampling Rate']))
                    else:
                        files_locator.at[index, 'Duration'] = ''
                    files_locator.at[index, 'File Type'] = "matRawData:" + str(source_data[0])

                elif isinstance(source_data_3d[1], ndarray):
                    files_locator.at[index, 'Sampling Rate'] = str(sampling_rate[1]).strip("HhZz")

                    if isinstance(channel_name[1], ndarray):
                        # If channel_name is ndarray, ensure correct processing and conversion to string
                        print(','.join(str(x) for x in channel_name[1]))
                        files_locator.at[index, 'Channel Names'] = ','.join(str(x) for x in channel_name[1])

                    files_locator.at[index, 'Number of Channels'] = str(len(channel_name[1]))
                    files_locator.at[index, 'Data Shape'] = str(source_data_3d[1].shape)
                    if is_numeric(files_locator.at[index, 'Sampling Rate']):
                        files_locator.at[index, 'Duration'] = str(
                            max(source_data_3d[1].shape) / float(files_locator.at[index, 'Sampling Rate']))
                    else:
                        files_locator.at[index, 'Duration'] = ''
                    files_locator.at[index, 'File Type'] = "matEpochData:" + str(source_data_3d[0])
                else:
                    # "No data detected in file {file_path}. Skipping."
                    pass
            else:
                # File {file_path} is below the size threshold. Skipping.
                pass
    return files_locator


def _find_variables_by_condition(data, condition_func, max_depth=5, max_width=5, debug=False):
    satisfying_variables = []
    _search_data(data, '', condition_func, satisfying_variables, 0, max_depth, max_width, debug=debug)
    if len(satisfying_variables) == 0:
        return "unknown", ''
    else:
        return satisfying_variables[0]


def _condition_source_data(var_path, var_value):
    if isinstance(var_value, ndarray) and var_value.ndim == 2 and var_value.nbytes > 5 * 1024 * 1024:
        return True
    return False


def _condition_source_data_3d(var_path, var_value):
    if isinstance(var_value, ndarray) and var_value.ndim == 3 and var_value.nbytes > 5 * 1024 * 1024:
        return True
    return False


def _condition_sampling_rate(var_path, var_value):
    """
    Condition function that checks if the variable path contains 'fs' or 'fre'.

    Parameters:
        var_path (str): The path of the variable.
        var_value: The value of the variable (unused in this condition).

    Returns:
        bool: True if 'fs' or 'fre' is in the variable path, False otherwise.
    """
    var_path = var_path.lower()
    return 'fs' in var_path or 'fre' in var_path


def _condition_sampling_channel_name(var_path, var_value):
    """
    Condition function that checks if the variable path contains 'chan'.

    Parameters:
        var_path (str): The path of the variable.
        var_value: The value of the variable (unused in this condition).

    Returns:
        bool: True if 'chan' is in the variable path, False otherwise.
    """
    var_path = var_path.lower()
    return ('chan' in var_path or 'chname' in var_path or 'clab' in var_path) and (
        isinstance(var_value, ndarray)) and var_value.shape[0] > 2


def _search_data(data, path, condition_func, satisfying_variables, current_depth=0, max_depth=5, max_width=5,
                 ignore_keys=None, debug=False):
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
