from eegunity.unifieddataset import UnifiedDataset
from eegunity.modules.parser import get_data_row
from eegunity.utils.pipeline import Pipeline
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
input_path = r'../../processed_raweeg_v2/bcic_iv_2a'  # Dataset directory
domain_tag = "demo-tag"      # Domain tag for marking the dataset
output_path = r'../../EEGUnity_CI/saving-h5Dataset'  # Output path for the h5Dataset
cache_path = r"../../EEGUnity_CI/saving-dataset-by-pipeline"  # Cache path for batch processing


# Define the processing function using the Pipeline for simplification
def app_func(row, output_dir):
    """
    Modify this function to adjust the processing steps.
    Processes each row of EEG data and saves the processed file.
    """
    pipeline = Pipeline(functions=[
        lambda mne_raw: mne_raw.pick_types(eeg=True, ecg=False),  # Remove non-EEG channels
        # lambda mne_raw: mne_raw.filter(l_freq=0.1, h_freq=45),  # Apply bandpass filter (0.1-45 Hz)
        # lambda mne_raw: mne_raw.notch_filter(freqs=45),  # Apply notch filter (45 Hz)
        # lambda mne_raw: mne_raw.resample(sfreq=200),  # Resample to 200 Hz
        # normalize_mne   # Custom normalization function
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
unified_dataset_process.eeg_batch.export_h5Dataset(output_path, name=domain_tag+"T3")