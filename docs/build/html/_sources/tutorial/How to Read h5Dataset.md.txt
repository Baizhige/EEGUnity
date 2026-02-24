# Reading EEGUnity HDF5 Files: A Guide

## 1. Introduction
This tutorial provides a step-by-step guide for reading EEG data stored in HDF5 files exported by the EEGUnity library (`UnifiedDataset.EEGBatch.export_h5Dataset`).
You'll learn what HDF5 files are, why certain attributes are stored redundantly, and how to read the data in Python.

## 2. What is HDF5?
HDF5 (Hierarchical Data Format version 5) is a widely-used file format for storing large amounts of structured data.
With a tree-like structure, HDF5 allows for grouping data and storing metadata (attributes) alongside datasets.
To learn more, visit the [HDF Group official website](https://www.hdfgroup.org/solutions/hdf5/).

## 3. Understanding Data Saving in HDF5
In EEGUnity files, HDF5 is used to store both EEG data and related metadata. Common attributes such as `rsFreq` (sampling frequency) and `chOrder` (channel order) are stored in two places:
1) as HDF5 attributes for easy access, and 2) within the `info` dataset for more details..
This redundancy aims to ensure interoperability and ease of data access across project [LaBraM](https://github.com/935963004/LaBraM)

## 4. Sample Script to Read HDF5 Files

Below is an example script for reading HDF5 files created by EEGUnity. This script opens the file, retrieves EEG data, and extracts important attributes like `rsFreq` and `chOrder`. It also reads and parses the `info` dataset, where additional metadata is stored, such as events.

```python
import h5py
import pickle

# Define the file path (replace 'your/file/path.hdf5' with your actual path)
file_path = 'your/file/path.hdf5'

# Open the HDF5 file and read specified fields
with h5py.File(file_path, 'r') as f:
    # Retrieve all groups
    group_names = list(f.keys())

    for i, grp_name in enumerate(group_names):
        print(f"\n=== Group {i} - {grp_name} ===")
        grp = f[grp_name]

        # Read EEG dataset
        if 'eeg' in grp:
            eeg_data = grp['eeg'][:]
            print(f"EEG Data Shape: {eeg_data.shape}")
        else:
            print("EEG dataset not found.")

        # Get the EEG dataset
        dset = grp['eeg']

        # Read 'rsFreq' attribute (sampling rate)
        if 'rsFreq' in dset.attrs:
            rs_freq = dset.attrs['rsFreq']
            print(f"Sampling Rate (rsFreq): {rs_freq}")
        else:
            print("Sampling rate (rsFreq) attribute not found.")

        # Read 'chOrder' attribute (channel order)
        if 'chOrder' in dset.attrs:
            ch_order = dset.attrs['chOrder']
            print(f"Channel Order (chOrder): {ch_order}")
        else:
            print("Channel order (chOrder) attribute not found.")

        # Read and parse the 'info' dataset
        if 'info' in grp:
            info_array = grp['info'][()]
            info_bytes = info_array.tobytes()
            raw_info = pickle.loads(info_bytes)
            print(raw_info)
        else:
            print("Info dataset not found.")

        print("\n" + "=" * 30)
```

**Note:** Replace `'your/file/path.hdf5'` with the path to your HDF5 file.

---

This script provides a basic framework for loading EEG data and metadata from HDF5 files. It prints key information from each group within the file, including EEG data shape, sampling rate (`rsFreq`), channel order (`chOrder`), and other metadata stored in `info`. You can expand or modify the script to suit specific analysis needs.

