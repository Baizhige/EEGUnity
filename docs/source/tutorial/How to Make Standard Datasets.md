# Tutorial: Creating a Standardized FIF Dataset using EEGUnity
## Prerequisites
Before using this script, you should have:
1. **Familiarity with Python syntax**  
   - Basic understanding of Python programming concepts and structures.

2. **Knowledge of the MNE-Python library (suggested)**  
   - Experience in using MNE-Python for EEG data processing and analysis.

3. **Understanding of EEG paradigms (suggested)**  
   - Familiarity with common EEG experimental paradigms and related concepts.

3. **Time Consuming**  
   - 30 mins

## Purpose 
This tutorial is made for developers to create a `make_fif_xxxx` script (below marked as `standard script`)using the `EEGUnity` library. The script standardizes raw EEG datasets into the FIF format for following any unified processing. It ensures that: 
- **Events** are stored as MNE annotations, with descriptions consistent with the original dataset. 
- **Subject Information** is embedded in the `mne.io.raw.info['description']` field. 
- **Channel Information** is updated if the original data does not sufficiently reflect electrode positions. 
This document is detailed and beginner-friendly, making it accessible even to those without prior knowledge of EEG or programming. Below are the steps to create `standard script`.
## 🚀 Step 0: Prepare a Python Environment

1. **Create a Python environment** with Python version **≥ 3.6**.
   - You can use `venv` or `conda` to manage the environment.
   
2. **Install the `EEGUnity` library** by running the following command in your terminal:
   ```bash
   pip install eeginity

## Step 1 Prepare a Python Project
1. Create a Python project and name it: `standard_script_projects`
2. Create a Python package named `setting` in the root directory of project and create a python file name `path_variable.py` inside the `setting` package
3. Create a folder in the root directory of the project, name `stage1_locator`, for storing locator files. 
4. Create a folder in the root directory of the project
5. Create a folder named `original_raweeg` in any accessible path, to save original EEG datasets (for dataset downloasd).
6. Create a folder named `standard_raweeg` in any accessible path, to save processed standard EEG datasets (folder for storing processed dataset).
7. Download any datasets and unzip EEG datasets in `orignal_raweeg`, such as [BCI Competition IV 2a](https://www.bbci.de/competition/iv/#dataset2a)
8. Define path variable on `path_variable.py`, like
    ```python
    # Input path for each dataset
    bcic_iv_2a_path = r"path/to/original_raweeg/bcic_iv_2a"
    bcic_iv_2b_path = r"path/to/original_raweeg/bcic_iv_2b" # add more path if needed
   
    # output directory for stage 1
    stage1_locator = r"path/to/stage1_locator/" # folder to save stage 1 locator
    stage1_output_dir = r"path/to/standard_raweeg"  # folder to save processed datasets
    ```

## Step 2 Make Standard Script for Datasets
Create a Python file, name `make_fif_xxxx.py` in the root directory of the project. Then, make it by following instrutions.
The script performs the following tasks: 
1. **Parameter Settings**:   
   Define input and output paths, domain tag, and cache usage. For example: 
   ```python 
   # Parameter settings 
   import os
   import setting.path_variable as pv # this is additional python file which stores all path variable
   input_path = pv.bcic_iv_2a_path  # Dataset directory 
   domain_tag = 'bcic-iv-2a'        # Domain tag for marking the dataset 
   output_path = os.path.join(pv.stage1_output_dir, 'bcic_iv_2a')  # Output path 
   use_cache = False 
   ```
   *Note: Avoid absolute paths. Only specify basic parameters.* 
2. **Loading the Dataset**:   
   The script uses the `UnifiedDataset` class to load the dataset. If a locator file exists and caching is enabled, it reuses that file; otherwise, it creates a new locator file. 
   ```python
   locator_path = os.path.join(pv.stage1_locator, os.path.basename(input_path)+".csv")
   if os.path.exists(locator_path) and use_cache:
       unified_dataset = UnifiedDataset(locator_path=locator_path)
   else:
       unified_dataset = UnifiedDataset(dataset_path=input_path, domain_tag=domain_tag)
       unified_dataset.save_locator(locator_path)
   ```
3. **Processing Each Data Row**:   
   A function `app_func` is defined to: 
   - Load the raw EEG data using `get_data_row`, a key function in `EEGUnity`. 
   - Extract the subject ID (e.g., from the file path) and update the subject information (such as age, gender, etc.) in `mne_raw.info['description']`. For convenience, you can check the dataset folder beforehand and store the information in a Python dictionary to simplify loading it within the code. 
   - Rename channels to match the standard system (e.g., 10-20 system). 
   - **Events Handling**:   
     - **⚡ Extract events from the dataset folder** and convert them back to annotations using `mne.annotations_from_events`.  
     **_Note:_** This step is **the most important** and **time-consuming** when creating `standard script`. 🚨
     For more details on MNE annotations, please read the [MNE-Python Annotations Documentation](https://mne.tools/stable/auto_tutorials/plot_annotations.html) :contentReference[oaicite:0]{index=0} carefully.  
   - Handle irregularities, such as nonstandard data (e.g., file names and event annotations), directly within the script.  
     This ensures that users can run the `standard script` immediately after downloading the datasets without additional modifications.
4. **Saving the Processed Data**:   
   After processing, the EEG data is saved as a FIF file. The output filename is modified (by appending `_raw.fif`) to conform to MNE naming conventions. 
5. **Batch Processing**:   
   Finally, the script applies `app_func` to all data rows (filtered to only those marked as `Completed`), processing the entire dataset in batch. 
## Detailed Explanation of the Code 
Below is the full sample script for standardizing the `bcic_iv_2a` dataset, you can copy it in your project and modify it based on your dataset: 
```python 
import scipy.io as scio
from eegunity.unifieddataset import UnifiedDataset
from eegunity.modules.parser import get_data_row
import numpy as np
import os
import mne
import setting.path_variable as pv
import json

subject_dict = {
    'A01': {'gender': 'female', 'age': 22},
    'A02': {'gender': 'female', 'age': 24},
    'A03': {'gender': 'male', 'age': 26},
    'A04': {'gender': 'female', 'age': 24},
    'A05': {'gender': 'male', 'age': 24},
    'A06': {'gender': 'female', 'age': 23},
    'A07': {'gender': 'male', 'age': 25},
    'A08': {'gender': 'male', 'age': 23},
    'A09': {'gender': 'male', 'age': 17}
}

# Parameter settings
input_path = pv.bcic_iv_2a_path  # Dataset directory
domain_tag = "bcic-iv-2a"      # Domain tag for marking the dataset
output_path = os.path.join(pv.stage1_output_dir, "bcic_iv_2a")  # Output path
use_cache = False

def app_func(row, output_dir):
    # Load the MNE raw data
    mne_raw = get_data_row(row)
    subject_id = os.path.basename(row['File Path'])[:3]
    description_dict = {
        "original_description": mne_raw.info['description'],
        "eegunity_description": {
            "amplifier": "unknown",
            "cap": "Ag/AgCl",
            "age": subject_dict[subject_id]['age'],
            "sex": subject_dict[subject_id]['gender'],
            "handedness": "unknown"
        }
    }
    mne_raw.info['description'] = json.dumps(description_dict)

    # Rename the channels to the 10-20 system, commonly used for 64 electrode positions
    mne_raw.rename_channels({
        'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz',
        'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3',
        'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
        'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz',
        'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-15': 'Pz',
        'EEG-16': 'P2', 'EEG-Pz': 'POz'
    })

    montage = mne.channels.make_standard_montage('standard_1020')
    mne_raw.info.set_montage(montage, on_missing='ignore')
    mne_raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})

    # Event ID mapping
    event_id = {
        'Rejected trial': 1,
        'Eye movements': 2,
        'Idling EEG (eyes open)': 3,
        'Idling EEG (eyes closed)': 4,
        'Start of a new run': 5,
        'Start of a trial': 6,
        'Cue onset left (class 1)': 7,
        'Cue onset right (class 2)': 8,
        'Cue onset foot (class 3)': 9,
        'Cue onset tongue (class 4)': 10
    }

    # Extract events and original event IDs
    events, original_event_id = mne.events_from_annotations(mne_raw)

    # Update events based on new event_id mapping
    for event_desc, new_id in event_id.items():
        if event_desc in original_event_id:
            events[events[:, 2] == original_event_id[event_desc], 2] = new_id

    # Check if the file name ends with 'E' and construct the .mat file path
    file_base, file_ext = os.path.splitext(row['File Path'])
    if file_base.endswith('E') and file_ext == '.gdf':
        mat_filepath = f"{file_base}.mat"
        if os.path.exists(mat_filepath):
            mat_data = scio.loadmat(mat_filepath)
            values_from_mat = mat_data[
                                  'classlabel'].flatten() + 6  # Replace 'data' with the correct key in your .mat file

            # Replace events where the last column is 7
            replacement_indices = np.where(events[:, -1] == 7)[0]
            if len(replacement_indices) >= len(values_from_mat):
                events[replacement_indices[:len(values_from_mat)], 2] = values_from_mat
            else:
                print(f"Warning: {mat_filepath} contains fewer values than needed for replacement.")

    # Convert modified events back to annotations
    event_desc = {value: key for key, value in event_id.items()}  # Convert event IDs back to descriptions
    annotations = mne.annotations_from_events(
        events=events,
        sfreq=mne_raw.info['sfreq'],
        event_desc=event_desc  # Mapping event codes to descriptions
    )
    mne_raw.set_annotations(annotations)  # Set new annotations to raw data

    # Save the processed EEG data to the output directory
    filename = os.path.basename(row['File Path'])  # Extract the file name
    # Modify the output filename to conform to MNE naming conventions
    output_filename = f"{filename[:-4]}_raw.fif"  # Assuming the original filename ends with '.gdf', remove the extension and add '_raw.fif'
    output_path = os.path.join(output_dir, output_filename)  # Define the output path
    mne_raw.save(output_path, overwrite=True)  # Save the file, overwriting if necessary

    return None


# 1. Load the dataset directory
locator_path = os.path.join(pv.stage1_locator, os.path.basename(input_path)+".csv")
if os.path.exists(locator_path) and use_cache:
    unified_dataset = UnifiedDataset(locator_path=locator_path)
else:
    unified_dataset = UnifiedDataset(dataset_path=input_path, domain_tag=domain_tag)
    unified_dataset.save_locator(locator_path)

# 2. Batch process EEG data
unified_dataset.eeg_batch.batch_process(
    con_func=lambda row: row['Completeness Check'] == 'Completed',  # Filter out 'Completed' data
    app_func=lambda row: app_func(row, output_path),  # Call the processing function and set output directory
    is_patch=False,  # No patching needed
    result_type=None  # No return type needed
)
```

## Key Points to Remember 
- **Annotations**: All events are stored as MNE annotations, ensuring consistency in event descriptions. For more details, refer to [MNE-Python Annotations](https://mne.tools/stable/auto_tutorials/plot_annotations.html) :contentReference[oaicite:1]{index=1}. 
- **Subject Information**: Subject details (e.g., age, gender) are embedded in the `info['description']` field. 
- **Channel Information**: Renaming channels and setting a standard montage guarantees accurate electrode positioning. 
- **Robustness**: Exception handling is built into the script (e.g., handling irregular file names and time annotations) so that users can process the dataset without manual directory modifications. 
- **Parameter Settings**: Only basic parameters (input_path, domain_tag, output_path, use_cache) need to be specified, with no absolute paths used. 
