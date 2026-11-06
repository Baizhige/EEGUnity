import mne


def compute_quality_score_mne(raw, ica_params=None):
    """Compute a data quality score with MNE artifact detection methods.

    Parameters
    ----------
    raw : mne.io.Raw
        EEG recording to score.
    ica_params : dict or None, optional
        Additional keyword arguments passed to :class:`mne.preprocessing.ICA`.

    Returns
    -------
    str
        Comma-separated quality score and EOG, EMG, and ECG component counts.

    Examples
    --------
    >>> score = compute_quality_score_mne(raw)  # doctest: +SKIP
    >>> len(score.split(","))
    4
    """
    if ica_params is None:
        ica_params = {}

    raw = raw.load_data()
    raw.filter(l_freq=0.1, h_freq=None, n_jobs=4)
    # Set ICA components to the number of EEG channels
    n_components = len([ch for ch in raw.info['chs'] if ch['kind'] == 2])  # 2 for EEG channels

    # Use ICA to detect components with artifacts
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter="auto", **ica_params)
    ica.fit(raw)

    # Check if EOG channels are present in the dataset
    eog_inds = []
    eog_channels = _artifact_channels(raw, "eog")
    if eog_channels:
        try:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_channels)
        except RuntimeError as error:
            print(f"Warning: {error}. EOG detection skipped.")

    # Check if EMG channels are present in the dataset
    emg_inds = []
    try:
        emg_inds, _ = ica.find_bads_muscle(raw)
    except RuntimeError as error:
        print(f"Warning: {error}. EMG detection skipped.")

    # Check if ECG channels are present in the dataset
    ecg_inds = []
    ecg_channels = _artifact_channels(raw, "ecg")
    if ecg_channels:
        try:
            ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=ecg_channels)
        except RuntimeError as error:
            print(f"Warning: {error}. ECG detection skipped.")

    # Combine all detected artifact indices
    artifact_inds = list(set(eog_inds + emg_inds + ecg_inds))
    artifact_ratio = len(artifact_inds) / n_components

    results = f"{1 - artifact_ratio}, {len(eog_inds)}, {len(emg_inds)}, {len(ecg_inds)}"
    return results


def _artifact_channels(raw, channel_type):
    """Return artifact channel names found by MNE type or name prefix.

    Parameters
    ----------
    raw : mne.io.Raw
        Recording whose channels are inspected.
    channel_type : {'eog', 'ecg'}
        Artifact channel type to locate.

    Returns
    -------
    list of str
        Unique matching channel names in recording order.

    Examples
    --------
    >>> _artifact_channels(raw, "eog")  # doctest: +SKIP
    ['EOG-left']
    """
    picks = mne.pick_types(raw.info, **{channel_type: True})
    typed_names = [raw.ch_names[index] for index in picks]
    named_channels = [
        name for name in raw.ch_names if channel_type in name.lower()
    ]
    return list(dict.fromkeys([*typed_names, *named_channels]))
