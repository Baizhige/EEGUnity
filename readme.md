# EEGUnity

## Overview

EEGUnity is a Python package designed for processing and analyzing `large-scale EEG data` efficiently. This guide will walk you through the Usage on Windows, macOS, and Linux. 
For more details on the motivation, concepts, and vision behind this project, please refer to the paper [EEGUnity: Open-Source Tool in Facilitating Unified EEG Datasets Towards Large-Scale EEG Model](https://arxiv.org/abs/2410.07196)

## Usage in Python Project
(Notes: This repository is planned for release on PyPI and the Conda community once a stable version is achieved.)
### Prerequisites

- Python 3.x
- Git
- See [requirements.txt](docs%2Frequirements.txt)

### Clone the Repository

First, clone the repository using Git:

#### Windows, macOS and Linux:
```bash
git clone https://github.com/Baizhige/EEGUnity/.git
```

### Usage in Python Projects

To use EEGUnity in your Python project, you will need to copy the `eegunity` folder to your project directory:

1. **Copy the `eegunity` folder** from the cloned repository to your Python project's folder:
   
   #### Windows:
   - Copy `EEGUnity\eegunity` into your project's directory.
   
   #### macOS and Linux:
   - Copy `EEGUnity/eegunity` into your project's directory.

2. Your project structure should resemble the following:
```
my_project/
│
├── eegunity/
│   └── __init__.py
└── your_script.py
```
3. Import the package in your Python project like this:

   ```python
   from eegunity.unifieddataset import UnifiedDataset
   ```

# Tutorial
1. How to Format Channel Name and Inspect Channel Data: [Click here to view the tutorial](./tutorial/How%20to%20Format%20Channel%20Name%20and%20Inspect%20Metadata.md)
2. How to Process Data and Export as h5Dataset: [Click here to view the tutorial](./tutorial/How%20to%20Process%20Data%20and%20Export%20as%20h5Dataset.md)
3. How to Read h5Dataset: [Click here to view the tutorial](./tutorial/How%20to%20Read%20h5Dataset.md)


# Project Documentation
You can view the project manual through the following link: [Click here to view the manual](https://eegunity.readthedocs.io/en/latest/)
