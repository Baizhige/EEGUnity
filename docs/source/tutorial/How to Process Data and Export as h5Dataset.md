# EEG Dataset Processing and h5Dataset Export

This tutorial provides two ways to process EEG data and export to h5Dataset with EEGUnity.

## Workflow A: Step-by-Step Pipeline

Use this workflow when you want clear intermediate outputs and easier debugging.

```python
import os
from eegunity import UnifiedDataset

INPUT_PATH = r"path/to/dataset"
DOMAIN_TAG = "demo_tag"

CACHE_BANDPASS = r"./cache/bandpass"
CACHE_NOTCH = r"./cache/notch"
CACHE_RESAMPLE = r"./cache/resample"
CACHE_NORM = r"./cache/normalize"
H5_OUTPUT_DIR = r"./output_h5"

for path in [CACHE_BANDPASS, CACHE_NOTCH, CACHE_RESAMPLE, CACHE_NORM, H5_OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)

ud = UnifiedDataset(dataset_path=INPUT_PATH, domain_tag=DOMAIN_TAG, num_workers=8)
ud.eeg_batch.sample_filter(completeness_check="Completed")
ud.eeg_batch.format_channel_names()

# 1) Band-pass filter
ud.eeg_batch.filter(
    output_path=CACHE_BANDPASS,
    filter_type="bandpass",
    l_freq=0.1,
    h_freq=75,
    get_data_row_params={
        "is_set_channel_type": True,
        "pick_types_params": {"eeg": True},
    },
)

# 2) Notch filter
ud.eeg_batch.filter(
    output_path=CACHE_NOTCH,
    filter_type="notch",
    notch_freq=50,
    get_data_row_params={
        "is_set_channel_type": True,
        "pick_types_params": {"eeg": True},
    },
)

# 3) Resample
ud.eeg_batch.resample(
    output_path=CACHE_RESAMPLE,
    resample_params={"sfreq": 200},
)

# 4) Compute mean/std and normalize
ud.eeg_batch.process_mean_std(domain_mean=False)
ud.eeg_batch.normalize(
    output_path=CACHE_NORM,
    norm_type="channel-wise",
    domain_mean=False,
)

# 5) Export HDF5
ud.eeg_batch.export_h5Dataset(output_path=H5_OUTPUT_DIR, name=DOMAIN_TAG)
```

## Workflow B: Quick Custom Pipeline with `batch_process`

Use this workflow when you want one custom function for all steps.

```python
import os
from eegunity import UnifiedDataset, get_data_row
from eegunity.utils.pipeline import Pipeline
from eegunity.utils.normalize import normalize_mne

INPUT_PATH = r"path/to/dataset"
DOMAIN_TAG = "demo_tag"
PIPELINE_CACHE = r"./cache/pipeline"
H5_OUTPUT_DIR = r"./output_h5"

os.makedirs(PIPELINE_CACHE, exist_ok=True)
os.makedirs(H5_OUTPUT_DIR, exist_ok=True)

ud = UnifiedDataset(dataset_path=INPUT_PATH, domain_tag=DOMAIN_TAG, num_workers=8)
ud.eeg_batch.sample_filter(completeness_check="Completed")
ud.eeg_batch.format_channel_names()


def process_row(row, output_dir):
    """Run a custom MNE pipeline for one EEG file and save a FIF result."""
    raw = get_data_row(row, is_set_channel_type=True)
    pipeline = Pipeline(
        functions=[
            lambda x: x.pick_types(eeg=True, ecg=False),
            lambda x: x.filter(l_freq=0.1, h_freq=75),
            lambda x: x.notch_filter(freqs=50),
            lambda x: x.resample(sfreq=200),
            normalize_mne,
        ]
    )
    raw_processed = pipeline.forward(raw)

    file_stem = os.path.splitext(os.path.basename(row["File Path"]))[0]
    output_path = os.path.join(output_dir, f"{file_stem}_pipeline_raw.fif")
    raw_processed.save(output_path, overwrite=True)
    return None


ud.eeg_batch.batch_process(
    con_func=lambda row: row["Completeness Check"] == "Completed",
    app_func=lambda row: process_row(row, PIPELINE_CACHE),
    is_patch=False,
    result_type=None,
    execution_mode="process",
)

# Load processed FIF files and export to h5Dataset.
ud_processed = UnifiedDataset(dataset_path=PIPELINE_CACHE, domain_tag=f"{DOMAIN_TAG}_pipeline")
ud_processed.eeg_batch.export_h5Dataset(output_path=H5_OUTPUT_DIR, name=f"{DOMAIN_TAG}_pipeline")
```

## Notes

- `export_h5Dataset` requires `output_path` to already exist.
- If the target `.hdf5` already exists, `export_h5Dataset` raises `FileExistsError`.
- Prefer `execution_mode="process"` for CPU-heavy transforms, and `execution_mode="thread"` for I/O-heavy tasks.
