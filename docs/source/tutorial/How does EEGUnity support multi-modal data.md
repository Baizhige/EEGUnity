# How to Support Multi-Modal Data in EEGUnity

EEGUnity supports unified channel naming and processing across multiple signal types.

Supported channel prefixes include:

- `eeg` for electroencephalography
- `eog` for electrooculography
- `emg` for electromyography
- `meg` for magnetoencephalography
- `ecg` for electrocardiography
- `stim` for stimulation/event channels
- `bio` for unmatched or generic biological channels
- `misc` for continuous label channels (for example `misc:reaction_time`)

EEGUnity also accepts explicit MNE channel type strings in locator entries (for example `seeg:LA1`, `ecog:G1`, `dbs:DBS1`, `fnirs_od:S1_D1_760`).
Legacy uppercase prefixes (for example `EEG`, `EOG`, `STIM`, `Unknown`) are still accepted for backward compatibility.

## Recommended Workflow

1. Load dataset metadata with `UnifiedDataset`.
2. Filter to completed records.
3. Format channel names into `<Type>:<Name>`.
4. Apply modality-specific processing with `batch_process`.

```python
from eegunity import UnifiedDataset

ud = UnifiedDataset(dataset_path="path/to/dataset", domain_tag="my_dataset")
ud.eeg_batch.sample_filter(completeness_check="Completed")
ud.eeg_batch.format_channel_names()
```

## About `misc:` Label Channels

`misc:` channels are useful when a dataset includes continuous labels (for example reaction time or score trajectories) that should stay aligned with EEG samples.

When resampling data, prefer EEGUnity helpers that preserve label semantics:

- `eegunity.utils.label_channel.resample_raw_with_labels`
- Methods that already call this helper internally, such as epoching and resampling paths in `eeg_batch`

## Custom Processing for Multi-Modal Workloads

For full control, use:

```python
ud.eeg_batch.batch_process(...)
```

In newer versions, `batch_process` supports `execution_mode`:

- `execution_mode='thread'` for I/O-heavy tasks
- `execution_mode='process'` for CPU-heavy tasks
- `execution_mode=None` for strict sequential execution

## Notes

- Channel naming consistency is the foundation for correct modality handling.
- If your dataset has custom channel conventions, format and validate the locator first.
- For dataset-specific metadata injection, see the kernel tutorial.
