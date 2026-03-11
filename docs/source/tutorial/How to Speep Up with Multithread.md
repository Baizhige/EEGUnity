# How to Speed Up EEGUnity with Built-in Parallelism

## 1. Introduction

EEGUnity supports parallel execution through:

- Global worker count: `UnifiedDataset(..., num_workers=...)`
- Batch backend selection: `batch_process(..., execution_mode=...)`

This tutorial explains when to use thread mode, process mode, or sequential mode.

## 2. Basic Configuration

```python
from eegunity import UnifiedDataset

ud = UnifiedDataset(
    dataset_path="your_dataset_root",
    domain_tag="your_domain_tag",
    num_workers=8,
)
```

- `num_workers=0` means sequential execution.
- `num_workers>0` enables parallel paths where supported.

## 3. Execution Modes in `batch_process`

`EEGBatch.batch_process` supports:

- `execution_mode='thread'`: recommended for I/O-bound tasks
- `execution_mode='process'`: recommended for CPU-bound tasks
- `execution_mode=None`: force sequential execution

Example:

```python
results = ud.eeg_batch.batch_process(
    con_func=lambda row: row["Completeness Check"] == "Completed",
    app_func=lambda row: row["File Path"],
    is_patch=False,
    result_type="value",
    execution_mode="thread",
)
```

## 4. Where Parallelism Is Commonly Applied

### 4.1 Parser Stage

When loading from `dataset_path`, parser steps can use `num_workers`, including:

- standard file scanning
- MAT/CSV parsing
- HDF5 EEGLAB `.set` fallback parsing
- BrainVision `.vhdr` sidecar fallback parsing
- WFDB header parsing

### 4.2 Batch Stage

Many heavy `eeg_batch` methods now choose backend explicitly, for example:

- filtering/resampling/normalization pipelines
- quality scoring
- hash calculation
- epoch workflows

Some write-heavy paths stay sequential intentionally to keep output consistency.

## 5. Choosing Worker Count

General guidance:

- Start at `os.cpu_count()`
- Reduce if memory pressure is high
- Increase slightly for storage/network bound workloads
- Benchmark on your own dataset

```python
import os
from eegunity import UnifiedDataset

ud = UnifiedDataset(
    dataset_path="your_dataset_root",
    domain_tag="your_domain_tag",
    num_workers=os.cpu_count(),
)
```

## 6. Practical Tips

- Use `execution_mode='process'` for CPU-heavy transforms.
- Use `execution_mode='thread'` for file I/O aggregation.
- Set `num_workers=0` during debugging for deterministic reproduction.
- Avoid nesting your own thread/process pools around EEGUnity batch calls.

## 7. Summary

Use `num_workers` to define available parallel workers and `execution_mode` to choose the right backend for each workload.
