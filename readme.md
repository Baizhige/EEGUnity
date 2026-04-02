<img src="./docs/source/_static/logo.png" alt="Project Logo" width="20%">

## Overview

EEGUnity is a Python package for unified parsing, preprocessing, and export of large-scale EEG datasets.

For project background, see the paper:
[EEGUnity: Open-Source Tool in Facilitating Unified EEG Datasets Towards Large-Scale EEG Model](https://arxiv.org/abs/2410.07196)

## Project Documentation

- Online manual (API + tutorials): [Read the docs](https://eegunity.readthedocs.io/en/latest/)

## Usage in Python Project

### 1. Create a Python Environment

Use Python 3.6 or higher. Python 3.13 is currently not supported due to dependency compatibility limits.

### 2. Install EEGUnity

```bash
pip install eegunity
```

### 3. Import EEGUnity

```python
from eegunity import UnifiedDataset
```

## Tutorial Navigation

### Core Tutorials

1. How does EEGUnity support multi-modal data: [Open](./docs/source/tutorial/How%20does%20EEGUnity%20support%20multi-modal%20data.md)
2. How to Format Channel Name and Inspect Metadata: [Open](./docs/source/tutorial/How%20to%20Format%20Channel%20Name%20and%20Inspect%20Metadata.md)
3. How to Make Standard Datasets: [Open](./docs/source/tutorial/How%20to%20Make%20Standard%20Datasets.md)
4. How to Process Data and Export as h5Dataset: [Open](./docs/source/tutorial/How%20to%20Process%20Data%20and%20Export%20as%20h5Dataset.md)
5. How to Read h5Dataset: [Open](./docs/source/tutorial/How%20to%20Read%20h5Dataset.md)

### Advanced Tutorials

1. How to Speed Up with Multithread: [Open](./docs/source/tutorial/How%20to%20Speep%20Up%20with%20Multithread.md)
2. How to Process Data Using Multiple Computers with EEGUnity: [Open](./docs/source/tutorial/How%20to%20Process%20Data%20Using%20Multiple%20Computers%20with%20EEGUnity.md)
3. How to Reading Rich Meta Data via Kernel: [Open](./docs/source/tutorial/How%20to%20Reading%20Rich%20Meta%20Data%20via%20Kernel.md)
4. How to Parse Non-standard Data Formats: [Open](./docs/source/tutorial/How%20to%20Parse%20Non-standard%20Data%20Formats.md)
5. How to Build File Hash and File Size Metadata: [Open](./docs/source/tutorial/How%20to%20Build%20File%20Hash%20and%20File%20Size%20Metadata.md)
