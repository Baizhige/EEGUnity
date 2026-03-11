# Working Across Multiple Computers with EEGUnity

This tutorial shows how two researchers share only the locator file, while keeping raw data on local disks.

## Scenario

- Researcher A and Researcher B both have a local copy of the same dataset.
- Paths are different on each machine.
- A sends a locator to B for reproducible processing.

## Step 1: Researcher A Generates Locator and Integrity Metadata

```python
from eegunity import UnifiedDataset

ud_a = UnifiedDataset(
    dataset_path=r"D:\eeg_data\my_dataset",
    domain_tag="my_dataset",
    num_workers=8,
)

ud_a.eeg_batch.sample_filter(completeness_check="Completed")
ud_a.eeg_batch.format_channel_names()

# Integrity metadata written to locator columns:
# - Source Hash (file-byte hash)
# - Data Hash (format-independent signal hash)
# - File Size (bytes)
ud_a.eeg_batch.get_file_hashes()
ud_a.eeg_batch.get_file_hashes(data_stream=True)
ud_a.eeg_batch.get_file_sizes()

ud_a.save_locator(r"./shared/my_dataset_locator.csv")
```

## Step 2: Researcher B Replaces Local Paths

```python
from eegunity import UnifiedDataset

ud_b = UnifiedDataset(locator_path=r"./shared/my_dataset_locator.csv", num_workers=8)

ud_b.eeg_batch.replace_paths(
    old_prefix=r"D:\eeg_data\my_dataset",
    new_prefix=r"E:\local_data\my_dataset",
)
```

`replace_paths` updates only the path prefix and keeps all other metadata unchanged.

## Step 3: Researcher B Recomputes and Verifies Consistency

```python
locator = ud_b.get_locator().copy()

# Preserve A-side references before recomputing hashes on B-side.
locator["Source Hash A"] = locator.get("Source Hash")
locator["Data Hash A"] = locator.get("Data Hash")
locator["File Size A"] = locator.get("File Size")
ud_b.set_locator(locator)

ud_b.eeg_batch.get_file_hashes()
ud_b.eeg_batch.get_file_hashes(data_stream=True)
ud_b.eeg_batch.get_file_sizes()

loc = ud_b.get_locator().copy()
loc["Source Hash Match"] = loc["Source Hash A"] == loc["Source Hash"]
loc["Data Hash Match"] = loc["Data Hash A"] == loc["Data Hash"]
loc["File Size Match"] = loc["File Size A"] == loc["File Size"]

mismatch = loc.loc[
    ~(loc["Source Hash Match"] & loc["Data Hash Match"] & loc["File Size Match"]),
    [
        "File Path",
        "Source Hash A",
        "Source Hash",
        "Data Hash A",
        "Data Hash",
        "File Size A",
        "File Size",
    ],
]

print("Mismatch rows:", len(mismatch))
print(mismatch.head())
```

## Notes

- Use `Data Hash` as the primary signal-level check when files may be repackaged in different container formats.
- `File Size == -1` indicates missing or inaccessible files on the current machine.
- Save the verified locator for downstream batch runs:

```python
ud_b.save_locator(r"./shared/my_dataset_locator_verified.csv")
```
