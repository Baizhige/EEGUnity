import numpy as np
import mne
from eegunity._modules.parser.eeg_parser import set_montage_any


def channel_align_raw(mne_raw, channel_order, min_matched_channel=1):
    """
    Aligns and orders the channels of an MNE Raw object according to a specified channel order.

    This function ensures that the channels in the raw MNE object are aligned and ordered
    according to the specified `channel_order`. If some channels from `channel_order`
    are missing in the raw data, they will be added with zero values and later interpolated.

    Parameters
    ----------
    mne_raw : mne.io.Raw
        The raw EEG/MEG data in an MNE Raw object.
    channel_order : list of str
        The desired order of channels.
    min_matched_channel : int, optional
        The minimum required number of matched channels, by default 1.

    Returns
    -------
    mne.io.Raw
        The modified raw object with channels aligned and missing channels interpolated.

    Raises
    ------
    ValueError
        If the number of matched channels is less than `min_matched_channel`.

    Notes
    -----
    - The function picks and reorders the matched channels to match `channel_order`.
    - If some channels from `channel_order` are missing in `mne_raw`, they are added as zero
      data channels and interpolated.
    - The missing channels are first marked as 'bad' before interpolation.

    Examples
    --------
    >>> import mne
    >>> raw = mne.io.read_raw_fif('sample_raw.fif', preload=True)
    >>> desired_order = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    >>> aligned_raw = channel_align_raw(raw, desired_order, min_matched_channel=5)
    """
    # Get existing channels in the raw object
    existing_channels = mne_raw.ch_names

    # Find the matched channels between the raw data and the desired channel order
    matched_channels = [ch for ch in channel_order if ch in existing_channels]

    # If the number of matched channels is less than the minimum required, raise an error
    if len(matched_channels) < min_matched_channel:
        raise ValueError(
            f"Error: Matched channels ({len(matched_channels)}) are less than the required minimum ({min_matched_channel})")


    # If there are missing channels in the raw data, handle them
    if len(matched_channels) < len(channel_order):
        missing_channels = [ch for ch in channel_order if ch not in existing_channels]
        mne_raw.load_data()

        # Create minimal info for the missing channels (only basic info required)
        missing_info = mne.create_info(missing_channels, sfreq=mne_raw.info['sfreq'], ch_types='eeg')

        # Create the missing channels with zero data
        missing_raw = mne.io.RawArray(np.zeros((len(missing_channels), len(mne_raw.times))), missing_info)

        # Add the missing channels to the raw data
        mne_raw.add_channels([missing_raw], force_update_info=True)

        # Use set_montage_any to add the montage (coordinates) for missing channels
        mne_raw = set_montage_any(mne_raw, verbose='CRITICAL')

        # Manually mark the missing channels as 'bads' so they can be interpolated
        mne_raw.info['bads'].extend(missing_channels)

        # Interpolate missing channels
        mne_raw.interpolate_bads(reset_bads=True, origin='auto', method=dict(meg="MNE", eeg="MNE", fnirs="nearest"))

    # Pick the matched channels and ensure the correct order
    mne_raw.pick_channels(matched_channels)
    mne_raw.reorder_channels(matched_channels)  # Ensures the channels are in the specified order

    return mne_raw
