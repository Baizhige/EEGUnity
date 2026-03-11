import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def _process_single_wfdb_file(file_path):
    """
    Process a single WFDB header file and return metadata dict or None.

    Parameters
    ----------
    file_path : str
        Path to the WFDB header file (.hea).

    Returns
    -------
    dict or None
        A dictionary containing extracted metadata, or None if the file cannot
        be processed or has no companion .dat data file.

    Examples
    --------
    >>> _process_single_wfdb_file("record.hea")  # doctest: +SKIP
    """
    import wfdb

    dat_path = os.path.splitext(file_path)[0] + '.dat'
    if not os.path.isfile(dat_path):
        return None

    try:
        record_name = os.path.splitext(file_path)[0]
        record = wfdb.rdheader(record_name)

        channel_names = list(record.sig_name)
        sampling_rate = record.fs
        n_samples = record.sig_len
        n_channels = record.n_sig

        # Deduplicate channel names: if duplicates exist, append _{n} suffix
        # (e.g. 256-channel SSVEP recordings that use generic 'EEG' for most
        # channels). Without dedup the completeness check always fails.
        seen = {}
        for i, name in enumerate(channel_names):
            if name in seen:
                seen[name] += 1
                channel_names[i] = f"{name}_{seen[name]}"
            else:
                seen[name] = 0

        result = {
            'File Type': 'wfdbData',
            'Sampling Rate': sampling_rate,
            'Channel Names': ','.join(channel_names),
            'Number of Channels': n_channels,
            'Data Shape': f'({n_channels}, {n_samples})',
            'Duration': n_samples / sampling_rate if sampling_rate else '',
        }
        return result
    except Exception as e:
        print(f"Error processing WFDB file {file_path}: {e}")
        return None


def process_wfdb_files(files_locator, num_workers=0):
    """
    Process WFDB header files and update a DataFrame with file details.

    Parameters
    ----------
    files_locator : pandas.DataFrame
        A DataFrame containing the metadata of files, including their file paths
        and other details. The column 'File Path' is expected to contain paths to
        the files. Only rows with 'File Type' equal to 'unknown' are processed.
    num_workers : int, optional
        Number of worker threads for parallel processing (default is 0, sequential).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with additional columns 'File Type', 'Sampling Rate',
        'Channel Names', 'Number of Channels', 'Data Shape', and 'Duration' for
        each eligible WFDB file. Files without a companion .dat file or that cannot
        be parsed are left unchanged.

    Examples
    --------
    >>> process_wfdb_files(locator_df, num_workers=2)  # doctest: +SKIP
    """
    eligible = []
    for index, row in files_locator.iterrows():
        file_path = row['File Path']
        file_type = row['File Type']
        if file_path.endswith('.hea') and file_type == 'unknown':
            eligible.append((index, file_path))

    if not eligible:
        return files_locator

    indices, file_paths = zip(*eligible)

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_single_wfdb_file, file_paths))
    else:
        results = [_process_single_wfdb_file(fp) for fp in file_paths]

    for idx, result in zip(indices, results):
        if result is not None:
            for key, value in result.items():
                files_locator.at[idx, key] = pd.NA if pd.isna(value) else str(value)

    return files_locator
