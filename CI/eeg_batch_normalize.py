import json
from eegunity.unifieddataset import UnifiedDataset

# Obtain base config from file
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']
CI_output_path = config['CI_output_path']

# Test the normalize function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)


    # Default parameters (sample-wise normalization)
    unified_dataset.eeg_batch.normalize(output_path=CI_output_path)
    print(f"Test with default parameters (sample-wise normalization) for {folder_name} completed.")

    # Normalize with min-max normalization
    unified_dataset.eeg_batch.normalize(output_path=CI_output_path, norm_type='sample-wise')
    print(f"Test with min-max normalization for {folder_name} completed.")

    # Normalize with z-score normalization
    unified_dataset.eeg_batch.normalize(output_path=CI_output_path, norm_type='channel-wise')
    print(f"Test with z-score normalization for {folder_name} completed.")

    # Testing normalization while skipping bad data (miss_bad_data=True)
    unified_dataset.eeg_batch.normalize(output_path=CI_output_path, miss_bad_data=True)
    print(f"Test with miss_bad_data=True for {folder_name} completed.")

    # Testing normalization with additional kwargs (custom parameters for `get_data_row`)
    unified_dataset.eeg_batch.normalize(output_path=CI_output_path, norm_type='sample-wise', custom_param1='value1', custom_param2='value2')
    print(f"Test with additional kwargs for {folder_name} completed.")

print("Successfully completed all normalize tests.")
