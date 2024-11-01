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
input_path = r'../../raweeg/figshare_meng2019'  # Dataset directory
domain_tag = "demp-tag"      # Domain tag for marking the dataset
output_path = r'../../EEGUnity_CI/saving-h5Dataset'  # Output path for the h5Dataset

# Intermediate processing cache paths
cache_bandpassfilter_path = r"../../EEGUnity_CI/saving-dataset-by-bandpass-filter"
cache_notchfilter_path = r"../../EEGUnity_CI/saving-dataset-by-notch-filter"
cache_resample_path = r"../../EEGUnity_CI/saving-dataset-by-resample"
cache_norm_path = r"../../EEGUnity_CI/saving-dataset-by-normalization"

# Locator file path for saving processed results
locator_norm_path = r"../../EEGUnity_CI/saving-locator/CI_locator_norm.csv"

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