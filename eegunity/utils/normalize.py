import numpy as np
import mne
from eegunity.utils.label_channel import misc_channel_indices, stim_channel_indices


def normalize_mne(mne_raw: mne.io.Raw) -> mne.io.Raw:
    """Normalize each non-misc/non-stim channel to zero mean and unit variance.

    This function processes data from an ``mne.io.Raw`` object and normalizes
    each eligible channel independently.

    Channels with MNE type ``misc`` and ``stim`` are excluded from
    normalization:

    - ``misc`` channels may carry continuous labels that should remain in
      original units.
    - ``stim`` channels contain integer trigger codes that must not be
      standardized.

    Parameters
    ----------
    mne_raw : mne.io.Raw
        Raw object containing EEG/MEG data.

    Returns
    -------
    mne.io.Raw
        The same raw object after in-place normalization.

    Notes
    -----
    Normalization is performed in place.

    Examples
    --------
    >>> raw = mne.io.read_raw_fif('sample_data.fif')
    >>> raw_normalized = normalize_mne(raw)
    >>> print(raw_normalized.get_data())
    """
    data = mne_raw.get_data()  # Get the raw data from the MNE Raw object

    # Identify channels to skip before normalization.
    skip_idx = set(misc_channel_indices(mne_raw)) | set(stim_channel_indices(mne_raw))
    non_misc_idx = np.array([i for i in range(len(mne_raw.ch_names)) if i not in skip_idx])

    if non_misc_idx.size == 0:
        return mne_raw

    eeg_data = data[non_misc_idx]
    mean = np.mean(eeg_data, axis=1, keepdims=True)
    std = np.std(eeg_data, axis=1, keepdims=True)
    data[non_misc_idx] = (eeg_data - mean) / std

    mne_raw._data = data  # Update the MNE Raw object with normalized data
    return mne_raw
