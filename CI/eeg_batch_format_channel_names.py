import json
from eegunity.unifieddataset import UnifiedDataset

# Obtain base config from file
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']
CI_output_path = config['CI_output_path']

# Test function format_channel_names
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    try:
        # Test the format_channel_names function
        unified_dataset.eeg_batch.format_channel_names()
        print(f"Test format_channel_names for {folder_name} completed successfully.")

        # Check if the 'Channel Names' column is correctly updated
        channel_names_column = unified_dataset.get_locator().loc[:, 'Channel Names']
        assert channel_names_column.notnull().all(), "Channel Names column contains null values."
        print(f"Channel Names column format verification for {folder_name} passed.")

    except KeyError as e:
        print(f"Test failed for {folder_name}: {str(e)}")

print("Successfully completed all format_channel_names tests.")
