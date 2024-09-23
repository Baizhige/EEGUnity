import json
from eegunity.unifieddataset import UnifiedDataset

# Obtain base config from file
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']
CI_output_path = config['CI_output_path']

# Define sample channel orders for testing
channel_order = ['C3', 'C4', 'Cz']

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)

    # Test with default parameters and channel order
    unified_dataset.eeg_batch.align_channel(output_path=CI_output_path, channel_order=channel_order)
    print(f"Test with channel order for {folder_name} completed.")

print("Successfully completed all align_channel tests.")
