import json
import os
import sys

# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from eegunity.unifieddataset import UnifiedDataset

# Obtain base config from file
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']
CI_output_path = config['CI_output_path']

# Create filter output directory if it doesn't exist
filter_output_path = os.path.join(CI_output_path, 'filter')
os.makedirs(filter_output_path, exist_ok=True)

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)

    # Default parameters (bandpass filter with no specific cutoff frequencies)
    unified_dataset.eeg_batch.filter(output_path=filter_output_path)
    print(f"Test with default parameters (bandpass) for {folder_name} completed.")

    # Low-pass filter
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.filter(output_path=filter_output_path, filter_type='lowpass', h_freq=40.0)
    print(f"Test with low-pass filter (h_freq=40.0) for {folder_name} completed.")

    # High-pass filter
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.filter(output_path=filter_output_path, filter_type='highpass', l_freq=1.0)
    print(f"Test with high-pass filter (l_freq=1.0) for {folder_name} completed.")

    # Bandpass filter with specific frequency range
    unified_dataset.eeg_batch.filter(output_path=filter_output_path, filter_type='bandpass', l_freq=1.0, h_freq=40.0)
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    print(f"Test with bandpass filter (l_freq=1.0, h_freq=40.0) for {folder_name} completed.")

    # Notch filter
    unified_dataset.eeg_batch.filter(output_path=filter_output_path, filter_type='notch', notch_freq=50.0)
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    print(f"Test with notch filter (notch_freq=50.0) for {folder_name} completed.")

    # Testing automatic adjustment of high frequency to fit the Nyquist frequency
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.filter(output_path=filter_output_path, filter_type='bandpass', l_freq=0.1, h_freq=256.0,
                                     auto_adjust_h_freq=True)
    print(f"Test with bandpass filter and auto-adjust h_freq for {folder_name} completed.")

    # Skipping bad data (miss_bad_data=True)
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.filter(output_path=filter_output_path, miss_bad_data=True)
    print(f"Test with miss_bad_data=True for {folder_name} completed.")

print("Successfully completed all filter tests.")
