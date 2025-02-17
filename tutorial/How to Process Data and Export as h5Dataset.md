# EEG Dataset Processing Tutorial

This section introduces two demos for processing EEG datasets using **EEGUnity** and exporting them as h5Dataset. The first demo follows a detailed step-by-step approach, while the second demo demonstrates a quicker processing method using a batch processing interface. Users can choose the appropriate method depending on their needs.

## Demo 1: Detailed Processing Workflow (`demo_make_h5Dataset.py`)

This demo illustrates how to process EEG datasets step by step and export them as h5Dataset. The process involves multiple IO operations, making it slower but more flexible for debugging. This method is suitable for users who want precise control over each processing step. The two demos presented in this tutorial allow for flexible EEG data processing depending on the user's needs. The first demo provides a detailed step-by-step approach for users who prefer fine control over each process, while the second demo offers a quicker and more streamlined batch processing method for users familiar with Python.


### Key Steps:
1. Load the dataset directory.
2. Filter out datasets marked as "Completed".
3. Remove non-EEG channels and standardize channel names.
4. Apply bandpass filtering (0.1-75 Hz).
5. Apply notch filtering at 50 Hz to remove powerline noise.
6. Resample the data to 200 Hz.
7. Normalize each EEG file by setting each channel’s mean to 0 and variance to 1.
8. Export the processed data as an h5Dataset.

### Code:
```python
from eegunity.unifieddataset import UnifiedDataset

# This script demonstrates step-by-step EEG processing and h5Dataset generation using EEGUnity.
# Main steps include:
# 1. Load the dataset directory
# 2. Filter out 'Completed' data
# 3. Standardize channel names and remove non-EEG channels (e.g., ECG)
# 4. Apply 0.1-75Hz bandpass filter
# 5. Apply 50Hz notch filter
# 6. Resample to 200Hz
# 7. Normalize data (mean = 0, variance = 1)
# 8. Export the processed data as an h5Dataset

# Parameter settings
input_path = r'path/to/dataset'  # Dataset directory
domain_tag = "demp-tag"      # Domain tag for marking the dataset
output_path = r'path/for/saving-h5Dataset'  # Output path for the h5Dataset

# Intermediate processing cache paths
cache_bandpassfilter_path = r"path/for/saving-dataset-by-bandpass-filter"
cache_notchfilter_path = r"path/for/saving-dataset-by-notch-filter"
cache_resample_path = r"path/for/saving-dataset-by-resample"
cache_norm_path = r"path/for/saving-dataset-by-normalization"

# Locator file path for saving processed results
locator_norm_path = r"path/for/saving-locator.csv"

# Code execution steps
# 1. Load the dataset directory
unified_dataset = UnifiedDataset(dataset_path=input_path, domain_tag=domain_tag)

# 2. Filter out 'Completed' status data
unified_dataset.eeg_batch.sample_filter(completeness_check='Completed')

# 3. Standardize channel names and prepare for removing non-EEG channels
unified_dataset.eeg_batch.format_channel_names()

# 4. Apply bandpass filtering (0.1-75Hz), select EEG channels
unified_dataset.eeg_batch.filter(output_path=cache_bandpassfilter_path, filter_type='bandpass', l_freq=0.1, h_freq=75,
                                 get_data_row_params = {"is_set_channel_type":True, "pick_types":{'eeg': True}})

# 5. Apply 50Hz notch filter (for powerline noise removal)
unified_dataset.eeg_batch.filter(output_path=cache_notchfilter_path, filter_type='notch', notch_freq=50)

# 6. Resample to 200Hz
unified_dataset.eeg_batch.resample(output_path=cache_resample_path, resample_params={"sfreq":200})

# 7-A. Calculate the mean and standard deviation for normalization
unified_dataset.eeg_batch.process_mean_std(domain_mean=False)

# 7-B. Normalize each channel (mean=0, variance=1)
unified_dataset.eeg_batch.normalize(cache_norm_path, norm_type='channel-wise')

# Save the locator file (optional)
unified_dataset.save_locator(locator_norm_path)

# 8. Export the processed data as an h5Dataset
unified_dataset.eeg_batch.export_h5Dataset(output_path, name=domain_tag, verbose=True)
```

## Demo 2: Quick Batch Processing (`demo_make_h5Dataset_quick.py`)

This demo uses **UnifiedDataset.eeg_batch.batch_process()** to streamline the entire process. This method is faster and suitable for users who are comfortable with Python programming. The processing steps are condensed into a single function, making it easier to manage larger datasets.

### Key Steps:
1. Load the dataset directory.
2. Filter out datasets marked as "Completed".
3. Remove non-EEG channels and standardize channel names.
4. Apply bandpass filtering (0.1-75 Hz).
5. Apply notch filtering at 50 Hz to remove powerline noise.
6. Resample the data to 200 Hz.
7. Normalize each EEG file by setting each channel’s mean to 0 and variance to 1.
8. Export the processed data as an h5Dataset.

### Code:

```python
from eegunity.unifieddataset import UnifiedDataset
from eegunity.modules.parser import get_data_row
from eegunity.utils.pipeline import Pipeline
from eegunity.utils.normalize import normalize_mne
import os

# This script demonstrates the use of batch processing to handle EEG datasets efficiently.
# Main steps include:
# 1. Load the dataset directory
# 2. Filter out 'Completed' data
# 3. Standardize channel names and remove non-EEG channels (e.g., ECG)
# 4. Apply 0.1-75Hz bandpass filter
# 5. Apply 50Hz notch filter
# 6. Resample to 200Hz
# 7. Normalize data (mean = 0, variance = 1)
# 8. Export the processed data as an h5Dataset

# Parameter settings
input_path = r'path/to/dataset'  # Dataset directory
domain_tag = "demo-tag"  # Domain tag for marking the dataset
cache_path = r"path/for/saving-dataset-by-pipeline"  # Cache path for batch processing
output_path = r"path/for/saving-h5Dataset"  # Output path for h5Dataset


# Define the processing function using the Pipeline for simplification
def app_func(row, output_dir):
    """
    Modify this function to adjust the processing steps.
    Processes each row of EEG data and saves the processed file.
    """
    pipeline = Pipeline(functions=[
        lambda mne_raw: mne_raw.pick_types(eeg=True, ecg=False),  # Remove non-EEG channels
        lambda mne_raw: mne_raw.filter(l_freq=0.1, h_freq=75),  # Apply bandpass filter (0.1-75 Hz)
        lambda mne_raw: mne_raw.notch_filter(freqs=50),  # Apply notch filter (50 Hz)
        lambda mne_raw: mne_raw.resample(sfreq=200),  # Resample to 200 Hz
        normalize_mne  # Custom normalization function
    ])
    mne_raw = get_data_row(row, is_set_channel_type=True)  # Get and set channel types, for later mne_raw.pick_types（）
    processed_mne_raw = pipeline.forward(mne_raw)  # Process data based on custom pipeline
    # Save the processed EEG data
    filename = os.path.basename(row['File Path'])  # Get the file name
    output_path = f"{output_dir}/{filename}_processed.fif"  # Define the output path
    processed_mne_raw.save(output_path, overwrite=True)  # Save the processed file
    return output_path


# 1. Load the dataset directory
unified_dataset = UnifiedDataset(dataset_path=input_path, domain_tag=domain_tag)

# 2. Standardize channel names
unified_dataset.eeg_batch.format_channel_names()

# 3. Batch process EEG data
unified_dataset.eeg_batch.batch_process(
    con_func=lambda row: row['Completeness Check'] == 'Completed',  # Filter out 'Completed' data
    app_func=lambda row: app_func(row, cache_path),  # Call the processing function and set output directory
    is_patch=False,  # No patching needed
    result_type=None  # No return type needed
)

# 4. Load the processed dataset and export as h5Dataset
unified_dataset_process = UnifiedDataset(dataset_path=cache_path, domain_tag=domain_tag)
unified_dataset_process.eeg_batch.export_h5Dataset(output_path, name=domain_tag, verbose=True)
```
