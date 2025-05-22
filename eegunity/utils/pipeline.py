class Pipeline:
    """
    Apply a list of functions sequentially to an input.

    The Pipeline class enables users to define and apply a sequence
    of transformations (functions) to input data.

    Attributes
    ----------
    functions : list of callable
        A list of functions to apply in order.
    Examples
    --------
    EEG processing pipeline using MNE:
    >>> import mne
    >>> def bandpass_filter(raw, l_freq, h_freq):
    ...     return raw.filter(l_freq=l_freq, h_freq=h_freq)
    >>> def notch_filter(raw, freqs):
    ...     return raw.notch_filter(freqs=freqs)
    >>> def resample(raw, sfreq):
    ...     return raw.resample(sfreq=sfreq)
    >>> # Define processing functions
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
        """
        Initialize the pipeline with a list of functions.

        Parameters
        ----------
        functions : list of callable
            Functions to apply sequentially to the input.
        """
        self.functions = functions

    def forward(self, X):
        """
        Apply all functions in the pipeline to the input data.

        Parameters
        ----------
        X : any
            The input data to be transformed.

        Returns
        -------
        any
            The transformed data after applying all functions.
        """
        for func in self.functions:
            X = func(X)
        return X
