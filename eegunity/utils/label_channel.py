"""Utilities for EEGUnity misc (continuous-label) channels.

EEGUnity uses the ``misc:`` channel name prefix together with MNE's built-in
``misc`` channel type to attach per-sample continuous signals to a
:class:`~mne.io.BaseRaw` object alongside the EEG data.  Typical use cases
include regression labels such as reaction time, inter-event gap, or any
other scalar value produced by a dataset-specific kernel for each epoch.

Channel naming convention
-------------------------
All EEGUnity label channels follow the ``misc:{task_name}`` pattern, for
example ``misc:reaction_time`` or ``misc:inter_event_gap``.  The MNE channel
type is always ``misc``.

This convention is consistent with EEGUnity's locator channel prefix system
(``eeg:``, ``eog:``, ``emg:``, ``ecg:``, ``stim:``; legacy uppercase forms are
also accepted) and is distinct from ``stim:`` channels in the following ways:

- Value type: ``stim`` channels carry integer trigger codes, while
  ``misc`` label channels carry continuous float values.
- Typical use: ``stim`` is for event onset or TTL-like pulses; ``misc`` is
  for per-sample regression targets.
- Resampling: MNE handles ``stim`` with nearest-neighbour logic, but applies
  ``resample_poly`` to ``misc`` channels unless EEGUnity's wrapper is used.
- ``filter()`` / ``ICA.fit()``: both channel types are excluded by default.
- ``events_from_annotations()`` does not directly consume either channel type.

Because MNE's :meth:`~mne.io.BaseRaw.resample` applies
``scipy.signal.resample_poly`` to *all* channels (misc included), label
channels must be resampled with nearest-neighbour interpolation to preserve
their original float values.  Always call :func:`resample_raw_with_labels`
instead of ``raw.resample()`` directly in any EEGUnity code path where
label channels may be present.

See Also
--------
:func:`resample_raw_with_labels` : Drop-in replacement for ``raw.resample()``.
:func:`is_misc_channel` : Predicate for identifying label channels.
:func:`misc_task_name` : Extract the task name from a label channel name.
"""

from typing import List

import numpy as np
import mne


MISC_CH_PREFIX: str = 'misc:'
"""Prefix string that identifies all EEGUnity misc (label) channels.

Every channel whose name starts with this prefix is treated as a continuous
label channel with MNE type ``misc``.
"""


def is_misc_channel(ch_name: str) -> bool:
    """Return ``True`` if *ch_name* is an EEGUnity misc (label) channel.

    The check is case-sensitive and matches the ``'misc:'`` prefix exactly.

    Parameters
    ----------
    ch_name : str
        Channel name to test.

    Returns
    -------
    bool
        ``True`` when *ch_name* starts with ``'misc:'``, ``False`` otherwise.

    Examples
    --------
    >>> is_misc_channel('misc:reaction_time')
    True
    >>> is_misc_channel('eeg:Fz')
    False
    """
    return str(ch_name).startswith(MISC_CH_PREFIX)


def is_misc_channel_in_raw(raw: mne.io.BaseRaw, ch_idx: int) -> bool:
    """Return ``True`` if channel index points to a misc label channel.

    The check primarily uses MNE channel type metadata and falls back to the
    EEGUnity ``misc:`` prefix for backward compatibility.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object.
    ch_idx : int
        Channel index.

    Returns
    -------
    bool
        ``True`` when channel type is ``misc`` or name starts with ``misc:``.

    Examples
    --------
    >>> # is_misc_channel_in_raw(raw, 0)  # doctest: +SKIP
    """
    try:
        if mne.channel_type(raw.info, ch_idx) == 'misc':
            return True
    except Exception:
        pass
    return is_misc_channel(raw.ch_names[ch_idx])


def is_stim_channel_in_raw(raw: mne.io.BaseRaw, ch_idx: int) -> bool:
    """Return ``True`` if channel index points to a stim channel.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object.
    ch_idx : int
        Channel index.

    Returns
    -------
    bool
        ``True`` when channel type is ``stim``.

    Examples
    --------
    >>> # is_stim_channel_in_raw(raw, 0)  # doctest: +SKIP
    """
    try:
        return mne.channel_type(raw.info, ch_idx) == 'stim'
    except Exception:
        return False


def misc_channel_indices(raw: mne.io.BaseRaw) -> List[int]:
    """Return indices of misc channels in a raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object.

    Returns
    -------
    list of int
        Indices of channels treated as misc labels.

    Examples
    --------
    >>> # idx = misc_channel_indices(raw)  # doctest: +SKIP
    """
    return [idx for idx in range(len(raw.ch_names)) if is_misc_channel_in_raw(raw, idx)]


def stim_channel_indices(raw: mne.io.BaseRaw) -> List[int]:
    """Return indices of stim channels in a raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object.

    Returns
    -------
    list of int
        Indices of channels with type ``stim``.

    Examples
    --------
    >>> # idx = stim_channel_indices(raw)  # doctest: +SKIP
    """
    return [idx for idx in range(len(raw.ch_names)) if is_stim_channel_in_raw(raw, idx)]


def misc_task_name(ch_name: str) -> str:
    """Extract the task name from a misc label channel name.

    Parameters
    ----------
    ch_name : str
        A channel name of the form ``'misc:{task_name}'``.

    Returns
    -------
    str
        The task name portion after the ``'misc:'`` prefix.

    Raises
    ------
    ValueError
        If *ch_name* does not start with ``'misc:'``.

    Examples
    --------
    >>> misc_task_name('misc:reaction_time')
    'reaction_time'
    """
    if not is_misc_channel(ch_name):
        raise ValueError(f"Not a misc label channel name: {ch_name!r}")
    return ch_name[len(MISC_CH_PREFIX):]


def resample_raw_with_labels(
    raw: mne.io.BaseRaw,
    sfreq: float,
    **kwargs,
) -> mne.io.BaseRaw:
    """Resample *raw*, applying nearest-neighbour interpolation to misc channels.

    MNE's :meth:`~mne.io.BaseRaw.resample` uses
    ``scipy.signal.resample_poly`` for all channels, which introduces
    low-pass filtering artefacts on the step-function signals typically
    stored in ``misc:`` label channels.  This function wraps the standard
    resample call and overwrites the resampled misc channel data with values
    obtained by nearest-neighbour interpolation, preserving the original float
    values exactly for samples that fall in the interior of constant regions.

    EEG, EOG, MEG and all other non-misc channels are resampled with the
    standard MNE pipeline and are not affected by this wrapper.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw object to resample.  Will be loaded into memory if not already
        preloaded.
    sfreq : float
        New sampling frequency in Hz.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`~mne.io.BaseRaw.resample` (e.g. ``npad``, ``window``).

    Returns
    -------
    mne.io.BaseRaw
        The resampled raw object (modified in-place).

    Notes
    -----
    If no ``misc:`` channels are present, this function is equivalent to
    calling ``raw.resample(sfreq, **kwargs)`` directly.

    The nearest-neighbour mapping is computed as::

        new_index[i] = round(i * old_n_times / new_n_times)

    which guarantees that samples in the interior of a constant label region
    are reproduced exactly regardless of the resampling ratio.

    Examples
    --------
    >>> raw = resample_raw_with_labels(raw, sfreq=256)
    """
    misc_ch_indices: List[int] = misc_channel_indices(raw)
    misc_ch_names: List[str] = [raw.ch_names[idx] for idx in misc_ch_indices]

    if not misc_ch_names:
        raw.resample(sfreq, **kwargs)
        return raw

    raw.load_data()

    # Record original misc channel data and sample count before resampling.
    old_n_times: int = raw.n_times
    old_misc_data = raw._data[misc_ch_indices, :].copy()  # (n_misc, old_n_times)

    # Standard resample: correct for EEG/MEG; misc data will be overwritten.
    raw.resample(sfreq, **kwargs)
    new_n_times: int = raw.n_times

    # Nearest-neighbour index mapping: new sample i -> nearest old sample.
    nn_indices = (
        np.round(np.linspace(0, old_n_times - 1, new_n_times))
        .astype(int)
        .clip(0, old_n_times - 1)
    )

    for i, ch_idx in enumerate(misc_ch_indices):
        raw._data[ch_idx] = old_misc_data[i, nn_indices]

    return raw
