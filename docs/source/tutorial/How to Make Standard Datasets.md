# Creating a Standardized FIF Dataset with EEGUnity

This tutorial shows how to build a dataset-specific `make_fif_xxxx.py` script that converts raw files into standardized FIF files.

## Prerequisites

- EEGUnity installed (`pip install eegunity`)
- Basic Python knowledge
- Optional but recommended: familiarity with MNE-Python

## Recommended Project Layout

```text
standard_script_project/
|-- locator/
|-- output_fif/
|-- make_fif_xxxx.py
```

## Step 1: Build Initial Locator

```python
import os
from eegunity import UnifiedDataset

INPUT_PATH = r"path/to/original_dataset"
DOMAIN_TAG = "my_dataset"
LOCATOR_PATH = r"./locator/my_dataset.csv"

os.makedirs(os.path.dirname(LOCATOR_PATH), exist_ok=True)

ud = UnifiedDataset(dataset_path=INPUT_PATH, domain_tag=DOMAIN_TAG)
ud.eeg_batch.sample_filter(completeness_check="Completed")
ud.eeg_batch.format_channel_names()
ud.save_locator(LOCATOR_PATH)
```

## Step 2: Define Dataset-Specific Conversion Logic

```python
import json
import os
import mne
from eegunity import UnifiedDataset, get_data_row

INPUT_PATH = r"path/to/original_dataset"
DOMAIN_TAG = "my_dataset"
OUTPUT_DIR = r"./output_fif"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ud = UnifiedDataset(dataset_path=INPUT_PATH, domain_tag=DOMAIN_TAG, num_workers=4)
ud.eeg_batch.sample_filter(completeness_check="Completed")
ud.eeg_batch.format_channel_names()


def make_fif_row(row, output_dir):
    """Convert one locator row into a standardized FIF file."""
    raw = get_data_row(row, is_set_channel_type=True)

    # Example: optional channel mapping and montage
    rename_map = {
        "EEG:CZ": "Cz",
        "EEG:FZ": "Fz",
    }
    valid_map = {k: v for k, v in rename_map.items() if k in raw.ch_names}
    if valid_map:
        raw.rename_channels(valid_map)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

    # Example: attach structured metadata in description
    raw.info["description"] = json.dumps(
        {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "dataset": DOMAIN_TAG,
                "pipeline": "make_fif_xxxx",
            },
        }
    )

    file_stem = os.path.splitext(os.path.basename(row["File Path"]))[0]
    output_path = os.path.join(output_dir, f"{file_stem}_raw.fif")
    raw.save(output_path, overwrite=True)
    return None


ud.eeg_batch.batch_process(
    con_func=lambda row: row["Completeness Check"] == "Completed",
    app_func=lambda row: make_fif_row(row, OUTPUT_DIR),
    is_patch=False,
    result_type=None,
    execution_mode="process",
)
```

## Step 3: Validate Converted Dataset

```python
from eegunity import UnifiedDataset

standard_ud = UnifiedDataset(dataset_path=r"./output_fif", domain_tag="my_dataset_fif")
print(standard_ud.get_locator().head())
```

## Notes

- Keep dataset-specific event parsing inside `make_fif_row`.
- If events are stored in sidecar files (`.mat`, `.csv`, `.tsv`), load them there and set MNE annotations before saving.
- Keep output filenames deterministic so downstream scripts are reproducible.
