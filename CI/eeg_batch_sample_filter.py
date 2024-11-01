import json
import os
import sys
# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from eegunity.unifieddataset import UnifiedDataset

# obtain base config from file
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)

    # Default parameters (should apply no filtering)
    unified_dataset.eeg_batch.sample_filter()
    print(f"Test with default parameters for {folder_name} completed.")

    # Filtering by channel_number
    unified_dataset.eeg_batch.sample_filter(channel_number=(32, 64))
    print(f"Test with channel_number=(32, 64) for {folder_name} completed.")

    # Filtering by sampling_rate
    unified_dataset.eeg_batch.sample_filter(sampling_rate=(128.0, 256.0))
    print(f"Test with sampling_rate=(128.0, 256.0) for {folder_name} completed.")

    # Filtering by duration
    unified_dataset.eeg_batch.sample_filter(duration=(1.0, 10.0))
    print(f"Test with duration=(1.0, 10.0) for {folder_name} completed.")

    # Filtering by completeness_check
    unified_dataset.eeg_batch.sample_filter(completeness_check='Completed')
    print(f"Test with completeness_check='Completed' for {folder_name} completed.")

    # Filtering by domain_tag
    unified_dataset.eeg_batch.sample_filter(domain_tag=folder_name)
    print(f"Test with domain_tag='{folder_name}' for {folder_name} completed.")

    # Filtering by file_type
    unified_dataset.eeg_batch.sample_filter(file_type='standard_data')
    print(f"Test with file_type='EDF' for {folder_name} completed.")

print("Successfully completed all sample_filter tests.")
