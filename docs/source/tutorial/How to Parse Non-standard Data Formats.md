# Parsing Non-standard Data Formats in EEGUnity

This tutorial covers non-standard file types supported by EEGUnity parser extensions.

## Supported Formats

EEGUnity can parse these non-standard sources during dataset scanning:

- MATLAB files: `.mat`
- HDF5 EEGLAB files: `.set` (stored as MATLAB v7.3/HDF5)
- CSV or TXT time-series tables: `.csv`, `.txt`
- WFDB records: `.hea` + `.dat`
- EDF content with non-standard extension: `.rec`
- BrainVision `.vhdr` with broken internal sidecar references (automatic patch fallback)

## Step 1: Build Locator with Parser Extensions Enabled

```python
from eegunity import UnifiedDataset

ud = UnifiedDataset(
    dataset_path=r"path/to/dataset",
    domain_tag="my_dataset",
    num_workers=8,
    min_file_size=0,  # include small CSV/TXT files
)

locator = ud.get_locator()
print(locator[["File Path", "File Type", "Completeness Check"]].head())
print(locator["File Type"].value_counts(dropna=False))
```

## Step 2: Inspect Specific File Types

```python
wfdb_rows = locator[locator["File Type"] == "wfdbData"]
csv_rows = locator[locator["File Type"] == "csvData"]
hdf5_set_rows = locator[locator["File Type"] == "eeglab_hdf5"]

print("WFDB rows:", len(wfdb_rows))
print("CSV/TXT rows:", len(csv_rows))
print("HDF5 .set rows:", len(hdf5_set_rows))
```

## Step 3: Load a Non-standard Row with `get_data_row`

```python
from eegunity import get_data_row

# Example: read the first available WFDB row
row = wfdb_rows.iloc[0]
raw = get_data_row(row, preload=False)

print("Channels:", raw.info["nchan"])
print("Sampling rate:", raw.info["sfreq"])
```

The same `get_data_row` API works for `.mat`, `csvData`, `eeglab_hdf5`, and `.rec` rows.

## Step 4: Batch Validate Readability

```python
def can_read(row):
    try:
        _ = get_data_row(row, preload=False)
        return "ok"
    except Exception as exc:
        return f"error: {type(exc).__name__}"

status = ud.eeg_batch.batch_process(
    con_func=lambda row: row["Completeness Check"] != "Unavailable",
    app_func=can_read,
    is_patch=True,
    result_type="value",
    execution_mode="thread",
)

ud.eeg_batch.set_metadata("Read Check", status)
print(ud.get_locator()[["File Path", "File Type", "Read Check"]].head())
```

## Notes

- For WFDB parsing, install `wfdb`.
- For HDF5 `.set`, install `h5py`.
- `min_file_size` mainly affects CSV/TXT scanning; set it to `0` when testing small demo files.
- BrainVision `.vhdr` sidecar mismatch is retried automatically with patched temporary headers.
