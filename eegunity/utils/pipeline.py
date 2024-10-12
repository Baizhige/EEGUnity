class Pipeline:
    """
    A pipeline that applies a list of functions sequentially to an input.

    The Pipeline class allows users to define a sequence of transformations
    (functions) that are applied to an input one after the other.

    Attributes:
        functions (list): A list of functions to be applied in sequence.

    Example usage (EEG processing):
        >>> import mne
        >>> def bandpass_filter(raw, l_freq, h_freq):
        ...     return raw.filter(l_freq=l_freq, h_freq=h_freq)
        >>> def notch_filter(raw, freqs):
        ...     return raw.notch_filter(freqs=freqs)
        >>> def resample(raw, sfreq):
        ...     return raw.resample(sfreq=sfreq)
        >>> # Load sample data
        >>> # raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_raw.fif', preload=True)
        >>> # Define processing functions for the pipeline
        >>> functions = [
        ...     lambda raw: bandpass_filter(raw, 0.1, 75),
        ...     lambda raw: notch_filter(raw, freqs=50),
        ...     lambda raw: resample(raw, sfreq=200)
        ... ]
        >>> # Initialize and apply the pipeline
        >>> pipeline = Pipeline(functions)
        >>> processed_raw = pipeline.forward(raw)
        >>> print(processed_raw.info['sfreq'])
    """

    def __init__(self, functions):
        self.functions = functions

    def forward(self, X):
        for func in self.functions:
            X = func(X)
        return X
