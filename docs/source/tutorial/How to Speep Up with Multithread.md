# Speeding Up EEGUnity Processing with Multithreading

## 1. Introduction

This tutorial shows you how to accelerate EEG data processing in EEGUnity using Python’s built-in multithreading support from the `concurrent` library. By processing EEG data in parallel, you can significantly reduce the time it takes to export large datasets.

If you're new to multithreading in Python, check out the [official Python documentation on `concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) or this helpful [Real Python guide on multithreading and multiprocessing](https://realpython.com/python-concurrency/).

---

## 2. When to Use Multithreading

Multithreading is useful when you need to perform many similar tasks that are not CPU-bound—for example, exporting EEG datasets that involve I/O operations. EEGUnity allows grouping datasets by domain, which makes it easy to split tasks for parallel processing.

---

## 3. Step-by-Step Guide

### Step 1: Group Dataset by Domain

Before you start parallel processing, you need to divide the dataset into smaller parts. EEGUnity provides a method to group the dataset by domain, using the `Domain Tag` column in your locator file.

> ⚠️ Make sure your locator file contains multiple entries with different values in the `Domain Tag` column.  
> If not, you can automatically split the dataset based on sampling rate and electrode configuration using `u_dataset.eeg_batch.auto_domain()`.
> Alternatively, you can group the data manually by assigning different values to the `Domain Tag` column in your locator file.
```python
from eegunity import UnifiedDataset

u_dataset = UnifiedDataset(locator_path=locator_path)
u_dataset_list = u_dataset.group_by_domain()  # Returns a list of UnifiedDataset instances
```

---

### Step 2: Define the Task Function

This function handles the specific task (For exmaple, `u_dataset_grouped.eeg_batch.export_h5Dataset()`) for each grouped dataset:

```python
def export_task(u_dataset_grouped):
    domain_tag = u_dataset_grouped.get_locator().iloc[0]['Domain Tag']
    print(f"[START] Exporting: {domain_tag}")

    u_dataset_grouped.eeg_batch.export_h5Dataset(output_path, name=domain_tag)

    print(f"[DONE ] Exported: {domain_tag}")
```

---

### Step 3: Run Tasks with `ThreadPoolExecutor`

Use a thread pool to run the export tasks in parallel:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# Limit the number of threads to avoid system overload
max_threads = min(8, len(u_dataset_list))

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(export_task, group) for group in u_dataset_list]

    for future in as_completed(futures):
        try:
            future.result()  # Wait for task to complete and raise exceptions if any
        except Exception as e:
            print(f"Task failed: {e}")
```

---

## 4. Summary

Using `ThreadPoolExecutor` with grouped EEGUnity datasets is a convenient way to speed up exporting. Make sure your task is I/O-bound or lightweight enough to benefit from multithreading.

If your tasks are CPU-intensive, consider using `ProcessPoolExecutor` instead.
