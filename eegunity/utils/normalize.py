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

    Raises
    ------
    ValueError
        If a signal channel contains NaN or infinite values. Label channels
        with type ``misc`` or ``stim`` are not subject to this check.

    Notes
    -----
    Normalization is performed in place. Finite constant signal channels are
    retained and mapped to all zeros because unit-variance scaling is undefined
    for them.

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

    signal_data = data[non_misc_idx]
    finite_by_channel = np.fromiter(
        (np.isfinite(channel).all() for channel in signal_data),
        dtype=bool,
        count=len(signal_data),
    )
    if not finite_by_channel.all():
        invalid_names = [
            mne_raw.ch_names[int(index)]
            for index in non_misc_idx[~finite_by_channel]
        ]
        raise ValueError(
            "cannot normalize channel(s) containing NaN or Inf: "
            f"{invalid_names}"
        )

    mean = np.mean(signal_data, axis=1, keepdims=True)
    std = np.std(signal_data, axis=1, keepdims=True)

    # A constant reference or auxiliary channel is valid input and should not
    # turn the entire recording into NaNs. The tolerance also catches tiny
    # floating-point residue left by filtering a constant channel. Mapping it
    # to zero is the well-defined limit of centering such a channel, while its
    # name, type, position, and sample count remain unchanged.
    lower = np.min(signal_data, axis=1, keepdims=True)
    upper = np.max(signal_data, axis=1, keepdims=True)
    scale = np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    constant = std <= np.finfo(signal_data.dtype).eps * scale
    signal_data -= mean
    np.divide(signal_data, std, out=signal_data, where=~constant)
    signal_data[constant[:, 0]] = 0.0
    data[non_misc_idx] = signal_data

    mne_raw._data = data  # Update the MNE Raw object with normalized data
    return mne_raw
