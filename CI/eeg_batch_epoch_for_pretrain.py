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

# Test function with different parameter combinations for epoch_for_pretraining
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)

    # Default parameters (no overlap, exclude bad data, no resample)
    unified_dataset.eeg_batch.epoch_for_pretraining(output_path=CI_output_path,
                                                    seg_sec=2.0)
    print(f"Test with default parameters for {folder_name} completed.")

    # Test with overlap
    unified_dataset.eeg_batch.epoch_for_pretraining(output_path=CI_output_path,
                                                    seg_sec=2.0, overlap=0.5)
    print(f"Test with overlap=0.5 for {folder_name} completed.")

    # Test with resampling
    unified_dataset.eeg_batch.epoch_for_pretraining(output_path=CI_output_path,
                                                    seg_sec=2.0, resample=100)
    print(f"Test with resample=100 Hz for {folder_name} completed.")

    # Test excluding bad data
    unified_dataset.eeg_batch.epoch_for_pretraining(output_path=CI_output_path,
                                                    seg_sec=2.0, exclude_bad=True)
    print(f"Test with exclude_bad=True for {folder_name} completed.")

    # Test with baseline correction
    unified_dataset.eeg_batch.epoch_for_pretraining(output_path=CI_output_path,
                                                    seg_sec=2.0, baseline=(None, 0.2))
    print(f"Test with baseline=(None, 0.2) for {folder_name} completed.")

    # Test skipping bad data
    unified_dataset.eeg_batch.epoch_for_pretraining(output_path=CI_output_path,
                                                    seg_sec=2.0, miss_bad_data=True)
    print(f"Test with miss_bad_data=True for {folder_name} completed.")

print("Successfully completed all epoch_for_pretraining tests.")
