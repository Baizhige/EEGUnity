# EEGUnity

## Overview

EEGUnity is a Python package designed for processing and analyzing EEG data efficiently. This guide will walk you through the Usage on Windows, macOS, and Linux.

## Usage in Python Project

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

2. Import the package in your Python project like this:

   ```python
   from eegunity.unifieddataset import UnifiedDataset
   ```

3. Your project structure should resemble the following:
```
my_project/
│
├── eegunity/
│   └── __init__.py
└── your_script.py
```

# Project Documentation
You can view the project manual through the following link: [Click here to view the manual](https://eegunity.readthedocs.io/en/latest/)
