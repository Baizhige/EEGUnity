import json
import os
import sys
# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from eegunity import UnifiedDataset

# Obtain base config from file
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']
CI_output_path = config['CI_output_path']

# Define sample channel orders for testing AFp6
channel_order = ['C3', 'C4', 'Cz']
channel_order_miss = ['C3', 'C4', 'Cz', 'AFp6']

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.format_channel_names()
    # Test with default parameters and channel order
    unified_dataset.eeg_batch.align_channel(output_path=CI_output_path+"/align_channel", channel_order=channel_order, get_data_row_params={"is_set_channel_type": True})
    print(f"Test with channel order for {folder_name} completed.")

for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.format_channel_names()
    # Test with default parameters and channel order
    unified_dataset.eeg_batch.align_channel(output_path=CI_output_path+"/align_channel", channel_order=channel_order_miss, get_data_row_params={"is_set_channel_type": True})
    print(f"Test with channel order for {folder_name} completed.")

print("Successfully completed all align_channel tests.")
