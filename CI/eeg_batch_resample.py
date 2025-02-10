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
CI_output_path = config['CI_output_path']+"/resample"

# Create filter output directory if it doesn't exist
resample_output_path = os.path.join(CI_output_path, 'resample')
os.makedirs(resample_output_path, exist_ok=True)

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)

    # Resample with specified sampling rate (downsample to 100 Hz)
    unified_dataset.eeg_batch.resample(output_path=resample_output_path, resample_params={"sfreq":100.0})
    print(f"Test with resample to 100 Hz for {folder_name} completed.")

    # Resample with specified sampling rate (upsample to 512 Hz)
    unified_dataset.eeg_batch.resample(output_path=resample_output_path, resample_params={"sfreq":512.0})
    print(f"Test with resample to 512 Hz for {folder_name} completed.")


print("Successfully completed all resample tests.")
