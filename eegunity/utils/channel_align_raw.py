import numpy as np
import mne
from eegunity.modules.parser.eeg_parser import set_montage_any
from eegunity.utils.label_channel import misc_channel_indices, stim_channel_indices


def channel_align_raw(mne_raw, channel_order, min_matched_channel=1):
    """
    Aligns and orders the channels of an MNE Raw object according to a specified channel order.

    This function ensures that the channels in the raw MNE object are aligned and ordered
    according to the specified `channel_order`. If some channels from `channel_order`
    are missing in the raw data, they will be added with zero values and later interpolated.

    ``misc`` label channels and ``stim`` trigger channels are temporarily
    removed before alignment so they do not interfere with EEG-specific
    operations (montage fitting, bad-channel interpolation). They are
    re-appended at the end of the channel list after alignment is complete.

    Parameters
    ----------
    mne_raw : mne.io.Raw
        The raw EEG/MEG data in an MNE Raw object.
    channel_order : list of str
        The desired order of channels. Should contain only channels to align
        (typically EEG). ``misc`` and ``stim`` channels are handled separately
        and must not be listed here.
    min_matched_channel : int, optional
        The minimum required number of matched channels, by default 1.

    Returns
    -------
    mne.io.Raw
        The modified raw object with channels aligned, missing channels interpolated,
        and preserved ``misc``/``stim`` channels appended at the end.

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
    - ``misc``/``stim`` channels are not interpolated and are not included in
      the alignment order.

    Examples
    --------
    >>> import mne
    >>> raw = mne.io.read_raw_fif('sample_raw.fif', preload=True)
    >>> desired_order = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    >>> aligned_raw = channel_align_raw(raw, desired_order, min_matched_channel=5)
    """
    # -----------------------------------------------------------------------
    # Step 1: Extract and remove preserved channels before EEG alignment.
    # They survive alignment unchanged and are re-appended at the end.
    # -----------------------------------------------------------------------
    preserve_idx = sorted(set(misc_channel_indices(mne_raw)) | set(stim_channel_indices(mne_raw)))

    if preserve_idx:
        mne_raw.load_data()
        preserve_names = [mne_raw.ch_names[idx] for idx in preserve_idx]
        preserve_data = mne_raw._data[preserve_idx, :].copy()
        preserve_types = [mne.channel_type(mne_raw.info, idx) for idx in preserve_idx]
        # Drop preserved channels so they don't interfere with EEG alignment.
        align_names = [ch for i, ch in enumerate(mne_raw.ch_names) if i not in preserve_idx]
        mne_raw.pick_channels(align_names)

    # -----------------------------------------------------------------------
    # Step 2: Standard EEG channel alignment.
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Step 3: Re-append preserved channels at the end of the channel list.
    # -----------------------------------------------------------------------
    if preserve_idx:
        preserved_info = mne.create_info(
            preserve_names,
            sfreq=mne_raw.info['sfreq'],
            ch_types=preserve_types,
        )
        preserved_raw = mne.io.RawArray(preserve_data, preserved_info, verbose=False)
        mne_raw.add_channels([preserved_raw], force_update_info=True)

    return mne_raw
