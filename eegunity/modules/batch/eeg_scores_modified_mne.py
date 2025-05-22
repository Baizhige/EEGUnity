import mne


def compute_quality_score_mne(raw, ica_params=None):
    """
    Compute a data quality score using MNE's built-in artifact detection methods.

    Parameters
    ----------
    raw : mne.io.Raw
        The EEG raw data.
    method : str
        The method to use for artifact detection. Options are "ica" and "maxwell".
    plot : bool
        Whether to plot artifact scores and diagnostics.

    Returns
    -------
    dict
        A dictionary containing the quality score, artifact ratio, and individual artifact counts.
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
    eog_channels = [ch for ch in raw.info['ch_names'] if 'eog' in ch.lower()]
    if eog_channels:
        eog_inds, eog_scores = ica.find_bads_eog(raw)

    # Check if EMG channels are present in the dataset
    emg_inds = []
    try:
        emg_inds, emg_scores = ica.find_bads_muscle(raw)
    except RuntimeError as e:
        print(f"Warning: {e}. EMG detection skipped.")

    # Check if ECG channels are present in the dataset
    ecg_inds = []
    ecg_channels = [ch for ch in raw.info['ch_names'] if 'ecg' in ch.lower()]
    if ecg_channels:
        try:
            ecg_inds, ecg_scores = ica.find_bads_ecg(raw)
        except RuntimeError as e:
            print(f"Warning: {e}. ECG detection skipped.")

    # Combine all detected artifact indices
    artifact_inds = list(set(eog_inds + emg_inds + ecg_inds))
    artifact_ratio = len(artifact_inds) / n_components

    results = f"{1 - artifact_ratio}, {len(eog_inds)}, {len(emg_inds)}, {len(ecg_inds)}"
    return results
