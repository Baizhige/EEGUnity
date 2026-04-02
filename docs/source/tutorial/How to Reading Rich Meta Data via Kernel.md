# EEGUnity Kernel Tutorial: Rich Metadata, `misc`, `stim`, and Annotations

This tutorial explains how to use EEGUnity kernels to inject dataset-specific metadata and channels in memory.

## 1. Design Principle

EEGUnity keeps **locator metadata as source of truth**:

- `format_channel_names()` standardizes locator channels as `channel_type:channel_name`.
- `get_data_row()` uses locator metadata to overwrite raw metadata at load time.
- Kernels are applied **after** locator-driven metadata patching.

This allows online metadata maintenance without modifying source files.

## 2. What a Kernel Can Do

A kernel can:

- add or update `raw.info["description"]`
- add or adjust multiple `misc` channels
- add or adjust multiple `stim` channels
- add/update annotations

Kernel interface:

```python
class SomeKernel:
    def apply(self, udataset, raw, row):
        ...
        return raw

KERNEL = SomeKernel()
```

## 3. Annotation vs `misc` vs `stim`

Use these three mechanisms for different semantics:

- `Annotations`: text labels mapped to time segments (`onset`, `duration`, `description`).
- `misc` channels: continuous values over time (for example probability density, reaction-time trajectory).
- `stim` channels: integer event codes over time (for example class sequence 1/2/3).

For a single scalar value for one segment, fill the covered segment in a `misc` channel.

## 4. Example Kernel with Multiple `misc` and `stim` Channels

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mne


def add_channel(raw: mne.io.BaseRaw, ch_name: str, ch_type: str, values: np.ndarray) -> mne.io.BaseRaw:
    """Append one channel to raw with explicit MNE channel type."""
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values.shape[0] != raw.n_times:
        raise ValueError("values length must equal raw.n_times")

    info = mne.create_info([ch_name], sfreq=raw.info["sfreq"], ch_types=[ch_type])
    ch_raw = mne.io.RawArray(values[np.newaxis, :], info, verbose=False)
    raw.add_channels([ch_raw], force_update_info=True)
    return raw


@dataclass
class ExampleKernel:
    KERNEL_ID: str = "example_rich_meta"

    def apply(self, udataset, raw: mne.io.BaseRaw, row):
        n = raw.n_times

        # misc channels (continuous signals)
        prob_density = np.linspace(0.1, 0.9, n, dtype=float)
        reaction_time = np.full(n, 0.42, dtype=float)
        raw = add_channel(raw, "prob_density", "misc", prob_density)
        raw = add_channel(raw, "reaction_time", "misc", reaction_time)

        # stim channels (integer codes)
        task_code = np.zeros(n, dtype=float)
        task_code[n // 4: n // 2] = 1
        task_code[n // 2: 3 * n // 4] = 2
        task_code[3 * n // 4:] = 3

        stage_code = np.zeros(n, dtype=float)
        stage_code[n // 3: 2 * n // 3] = 7

        raw = add_channel(raw, "task_code", "stim", task_code)
        raw = add_channel(raw, "stage_code", "stim", stage_code)

        # annotation segments (text semantics)
        ann = mne.Annotations(
            onset=[0.0, raw.times[n // 2]],
            duration=[2.0, 2.0],
            description=["trial_start", "feedback"],
        )
        raw.set_annotations(ann)

        return raw


KERNEL = ExampleKernel()
```

## 5. Binding and Running

```python
from eegunity import UnifiedDataset

ud = UnifiedDataset(
    dataset_path=r"path/to/dataset",
    domain_tag="my_dataset",
    kernel_spec=r"path/to/example_kernel.py",
)

# Parser path
raw0 = ud.eeg_parser.get_data(0)

# Batch path (kernel is also applied when loading row data in batch methods)
ud.eeg_batch.get_file_hashes(data_stream=True)
```

## 6. Channel Type Compatibility

EEGUnity standard prefixes are lowercase MNE-style (`eeg`, `eog`, `emg`, `ecg`, `meg`, `stim`, `misc`, `bio`) and it also accepts explicit MNE channel type strings in locator entries, for example:

- `seeg:LA1`
- `ecog:G1`
- `dbs:DBS1`
- `fnirs_od:S1_D1_760`
- `pupil:pupil_left`
- `misc:prob_density`
- `stim:task_code`

Legacy uppercase prefixes (`EEG`, `EOG`, `EMG`, `ECG`, `STIM`, `Unknown`) are accepted for backward compatibility.

## 7. Recommended Practice

- Use annotations for semantic event intervals.
- Use `stim` for integer-coded sequences.
- Use `misc` for continuous labels.
- Keep kernel logic dataset-specific and deterministic.
