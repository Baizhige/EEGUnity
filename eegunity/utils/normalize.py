import numpy as np
import mne
def normalize_mne(mne_raw: mne.io.Raw) -> mne.io.Raw:
    """
    Normalize each channel of the given MNE Raw object so that its mean is 0 and its standard deviation is 1.

    This function processes the data from the MNE Raw object and normalizes each channel independently.
    The mean of each channel will be set to 0, and the standard deviation will be set to 1,
    effectively standardizing the data across channels.

    Parameters:
    -----------
    mne_raw : mne.io.Raw
        An instance of the MNE Raw object containing EEG/MEG data to be normalized.

    Returns:
    --------
    mne.io.Raw
        The input MNE Raw object with its data normalized per channel.

    Notes:
    ------
    The normalization is done in place, meaning the original data in `mne_raw` is modified.

    Example:
    --------
    >>> raw = mne.io.read_raw_fif('sample_data.fif')
    >>> raw_normalized = normalize_mne(raw)
    >>> print(raw_normalized.get_data())
    """

    data = mne_raw.get_data()  # Get the raw data from the MNE Raw object
    mean = np.mean(data, axis=1, keepdims=True)  # Compute the mean of each channel
    std = np.std(data, axis=1, keepdims=True)  # Compute the standard deviation of each channel
    normalized_data = (data - mean) / std  # Perform normalization
    mne_raw._data = normalized_data  # Update the MNE Raw object with normalized data
    return mne_raw
