<img src="./docs/source/_static/logo.png" alt="Project Logo" width="20%">

## Overview

EEGUnity is a Python package designed for processing and analyzing `large-scale EEG data` efficiently. This guide will walk you through the Usage on Windows, macOS, and Linux. 
For more details on the motivation, concepts, and vision behind this project, please refer to the paper [EEGUnity: Open-Source Tool in Facilitating Unified EEG Datasets Towards Large-Scale EEG Model](https://arxiv.org/abs/2410.07196)

## Project Documentation
You can view the API Reference and Tutorial through the following link: [Click here to view the manual](https://eegunity.readthedocs.io/en/latest/)

## Usage in Python Project
### 1. Create a Python Environment
"Ensure you are using Python version 3.6 or higher. Due to compatibility issues with the latest version of NumPy, Python 3.13 is currently not supported.
### 2. Install EEGUnity via pip
Run the following command to install EEGUnity:
```bash
pip install eegunity
```

### 3. Import EEGUnity in Your Python Project
Use the following import statement to include the package:
```python
from eegunity import UnifiedDataset
```

## Tutorial
1. How does EEGUnity support multi-modal data: [Click here to view the tutorial](./docs/source/tutorial/How%20does%20EEGUnity%20support%20multi-modal%20data.md)
2. How to Format Channel Name and Inspect MetaData: [Click here to view the tutorial](./docs/source/tutorial/How%20to%20Format%20Channel%20Name%20and%20Inspect%20Metadata.md)
3. How to Make Standard Datasets: [Click here to view the tutorial](./docs/source/tutorial/How%20to%20Make%20Standard%20Datasets.md)
4. How to Process Data and Export as h5Dataset: [Click here to view the tutorial](./docs/source/tutorial/How%20to%20Process%20Data%20and%20Export%20as%20h5Dataset.md)
5. How to Process Data Using Multiple Computers with EEGUnity: [Click here to view the tutorial](./docs/source/tutorial/How%20to%20Process%20Data%20Using%20Multiple%20Computers%20with%20EEGUnity.md)
6. How to Read h5Dataset: [Click here to view the tutorial](./docs/source/tutorial/How%20to%20Read%20h5Dataset.md)
7. How to Speep Up with Multithread: [Click here to view the tutorial](./docs/source/tutorial/How%20to%20peep%20Up%20with%20Multithread.md)
