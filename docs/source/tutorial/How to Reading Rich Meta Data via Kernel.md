# EEGUnity Kernels: Dataset-Specific In-Memory Preprocessing

## 1. Introduction

EEGUnity provides a unified interface for parsing, preprocessing, and
managing EEG datasets. However, many public datasets are not fully
standardized:

-   Event markers may be stored in separate `.mat`, `.tsv`, or `.csv`
    files.
-   Subject metadata may exist in independent tables.
-   Channel naming conventions may vary across releases.
-   Folder structures may differ between mirrors or versions.

To address this variability **without duplicating EEG data**, EEGUnity
introduces the concept of external kernels.

A kernel is a dataset-specific, in-memory preprocessing plugin that runs
automatically when data is read.

------------------------------------------------------------------------

## 2. Why Use Kernels?

Traditional workflows often:

1.  Load raw data
2.  Run dataset-specific preprocessing scripts
3.  Export a new standardized dataset copy

This approach duplicates EEG arrays and complicates maintenance.

Kernels solve this by:

-   Running at read time
-   Updating `mne.io.Raw` objects in memory
-   Attaching metadata and annotations dynamically
-   Leaving the original dataset untouched

------------------------------------------------------------------------

## 3. How Kernels Work

When binding a kernel:

``` python
from eegunity import UnifiedDataset

ud = UnifiedDataset(
    dataset_path="/data/openneuro/ds005505",
    domain_tag="openneuro_ds005505",
    kernel_spec="/abs/path/openneuro_ds005505_kernel"
)

raw = ud.eeg_parser.get_data(0)
```

Internally:

1.  EEGUnity loads the Raw object.
2.  The external kernel is loaded dynamically.
3.  The system calls:

``` python
kernel.apply(udataset, raw, row)
```

If the kernel fails, EEGUnity emits a warning and returns the unmodified
raw.

------------------------------------------------------------------------

## 4. Kernel File Requirements

Each kernel file must:

1.  Be a single Python module
2.  Define exactly one object named:

``` python
KERNEL = YourKernelClass()
```

3.  Implement:

``` python
apply(udataset, raw, row) -> raw
```

One file equals one kernel. No suffix such as `:KERNEL` is required.

Valid kernel specifications:

-   File path (extension optional): "/abs/path/figshare_largemi_kernel"

-   Module import path: "my_private_kernels.figshare_largemi_kernel"

------------------------------------------------------------------------

## 5. Recommended Naming Convention

Kernel names should reflect the dataset source:

-   figshare_xxxx
-   openneuro_ds005505
-   kaggle_xxxx
-   bcic_iv_2a

Inside the kernel class:

``` python
KERNEL_ID = "figshare_largemi"
```

------------------------------------------------------------------------

## 6. Kernel Interface Specification

Required structure:

``` python
class SomeKernel:
    def apply(self, udataset, raw, row):
        ...
        return raw

KERNEL = SomeKernel()
```

Parameters:

-   udataset: dataset-level context
-   raw: loaded MNE Raw object
-   row: locator row (contains "File Path")

Return the modified raw object.

------------------------------------------------------------------------

## 7. Determining Dataset Root

If instantiated with dataset_path, use it directly.

If instantiated with locator_path only:

1.  Use udataset.get_shared_attr()\["dataset_path"\] if available.
2.  Otherwise compute common minimal prefix of all File Path entries.
3.  Fallback to directory of row\["File Path"\].

------------------------------------------------------------------------

## 8. Writing Robust Kernels

To support dataset variants:

-   Avoid hardcoded paths
-   Search recursively for participants or event files
-   Tolerate alternate column names
-   Handle missing metadata gracefully
-   Avoid assuming fixed folder structures

Focus on robust logic. EEGUnity handles exception safety.

------------------------------------------------------------------------

## 9. Minimal Kernel Template

``` python
from __future__ import annotations
import json
from dataclasses import dataclass
import mne

@dataclass
class ExampleKernel:
    KERNEL_ID: str = "source_name"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "source_name": self.KERNEL_ID
            },
        }
        raw.info["description"] = json.dumps(description_dict)
        return raw

KERNEL = ExampleKernel()
```

------------------------------------------------------------------------

## 10. Summary

Kernels allow EEGUnity to remain lightweight, avoid licensing issues,
and support diverse datasets through dynamic, in-memory preprocessing.
