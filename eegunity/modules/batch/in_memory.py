"""Run several Raw transforms in one load/save cycle."""

import hashlib
import os
from typing import Iterable, Sequence


def apply_raw_steps(raw, steps: Sequence):
    """Apply callables to one Raw object, left to right.

    A step may return a Raw object or mutate ``raw`` and return ``None``.
    The same helper can be passed to ``epoch_*_hdf5(..., pipeline=...)`` and
    ``export_h5Dataset(..., pipeline=...)`` so those writers do not reload
    the recording for each transform.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to transform.
    steps : sequence of callable
        Each callable accepts a Raw object.

    Returns
    -------
    mne.io.BaseRaw
        The recording after every step.
    """
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Iterable):
        raise TypeError("steps must be a sequence of callables.")
    steps = list(steps)
    if not steps:
        raise ValueError("steps must contain at least one callable.")
    for step in steps:
        if not callable(step):
            raise TypeError(f"Each step must be callable, got {type(step).__name__}.")
        updated = step(raw)
        if updated is not None:
            raw = updated
    return raw


def sync_saved_row(row, raw, file_path):
    """Copy signal fields from ``raw`` onto a locator row."""
    updated = row.copy()
    n_channels = int(raw.info["nchan"])
    n_times = int(raw.n_times)
    updated["File Path"] = file_path
    updated["File Type"] = "standard_data"
    updated["Data Shape"] = str((n_channels, n_times))
    updated["Channel Names"] = ", ".join(raw.ch_names)
    updated["Number of Channels"] = str(n_channels)
    updated["Sampling Rate"] = str(float(raw.info["sfreq"]))
    updated["Duration"] = str(float(raw.times[-1]) if n_times else 0.0)
    return updated


def output_fif_path(output_path: str, source_path: str, suffix: str) -> str:
    """Build a FIF path that stays unique when basenames collide."""
    file_name = os.path.splitext(os.path.basename(str(source_path)))[0]
    digest = hashlib.sha1(os.path.abspath(str(source_path)).encode("utf-8")).hexdigest()[:8]
    safe_suffix = str(suffix).strip() or "processed"
    return os.path.join(output_path, f"{file_name}_{digest}_{safe_suffix}_raw.fif")
