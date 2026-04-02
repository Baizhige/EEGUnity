# Reading EEGUnity HDF5 Files

This tutorial explains how to read `.hdf5` files exported by `UnifiedDataset.eeg_batch.export_h5Dataset`.

## HDF5 Layout Used by EEGUnity

For each EEG file, EEGUnity creates one HDF5 group at the root level:

- `<group_name>/eeg`: EEG array (`n_channels`, `n_samples`)
- `<group_name>/info`: pickled `mne.Info` bytes (`uint8` array)
- Attributes on `<group_name>/eeg`:
  - `rsFreq`: sampling rate
  - `chOrder`: channel order

## Example Script

```python
import h5py
import pickle

file_path = r"path/to/EEGUnity_export.hdf5"

with h5py.File(file_path, "r") as f:
    group_names = list(f.keys())
    print("Number of groups:", len(group_names))

    for i, grp_name in enumerate(group_names):
        grp = f[grp_name]
        print(f"\n=== Group {i}: {grp_name} ===")

        if "eeg" not in grp:
            print("Missing dataset: eeg")
            continue

        eeg_dset = grp["eeg"]
        eeg_data = eeg_dset[:]
        rs_freq = eeg_dset.attrs.get("rsFreq", None)
        ch_order = eeg_dset.attrs.get("chOrder", None)

        print("EEG shape:", eeg_data.shape)
        print("Sampling rate:", rs_freq)
        print("Channel order:", ch_order)

        if "info" in grp:
            info_bytes = grp["info"][()].tobytes()
            mne_info = pickle.loads(info_bytes)
            print("MNE info keys:", list(mne_info.keys())[:10])
        else:
            print("Missing dataset: info")
```

## Optional: Load Only Metadata

If you only need metadata without loading full EEG arrays:

```python
import h5py

file_path = r"path/to/EEGUnity_export.hdf5"

with h5py.File(file_path, "r") as f:
    for grp_name in f.keys():
        eeg_dset = f[grp_name]["eeg"]
        print(
            grp_name,
            eeg_dset.shape,
            eeg_dset.attrs.get("rsFreq", None),
        )
```

## Notes

- `info` is serialized with Python `pickle`; load only files from trusted sources.
- `chOrder` should be used together with EEG array rows when feeding models.
- If you need random access at scale, iterate by group and avoid loading all groups into memory at once.
