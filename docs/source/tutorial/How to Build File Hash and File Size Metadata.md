# Building File Hash and File Size Metadata

This tutorial shows how to add integrity metadata into the EEGUnity locator.

## Metadata Columns

- `Source Hash`: SHA-256 of raw file bytes (`get_file_hashes()`)
- `Data Hash`: SHA-256 of sampled EEG signal fingerprint (`get_file_hashes(data_stream=True)`)
- `File Size`: on-disk file size in bytes (`get_file_sizes()`)

## Step 1: Compute Integrity Metadata

```python
from eegunity import UnifiedDataset

ud = UnifiedDataset(
    dataset_path=r"path/to/dataset",
    domain_tag="my_dataset",
    num_workers=8,
)

ud.eeg_batch.sample_filter(completeness_check="Completed")
ud.eeg_batch.get_file_hashes()                 # Source Hash
ud.eeg_batch.get_file_hashes(data_stream=True) # Data Hash
ud.eeg_batch.get_file_sizes()                  # File Size

locator = ud.get_locator()
print(locator[["File Path", "Source Hash", "Data Hash", "File Size"]].head())
```

## Step 2: Find Potential Duplicates

```python
source_dup = locator[locator.duplicated("Source Hash", keep=False)]
data_dup = locator[locator.duplicated("Data Hash", keep=False)]

print("Source-level duplicates:", len(source_dup))
print("Signal-level duplicates:", len(data_dup))
```

Use `Data Hash` when files may be repackaged but still contain the same signal.

## Step 3: Find Missing/Unreadable Files

```python
missing_rows = locator[locator["File Size"].astype(float) < 0]
print("Missing or inaccessible files:", len(missing_rows))
print(missing_rows[["File Path", "File Size"]].head())
```

## Step 4: Save the Enriched Locator

```python
ud.save_locator(r"./locator/my_dataset_with_integrity.csv")
```

## Notes

- `Source Hash` can differ across formats for the same signal.
- `Data Hash` is more robust to channel order and minor representation differences.
- For very large datasets, run with `num_workers > 0` to speed up metadata generation.
