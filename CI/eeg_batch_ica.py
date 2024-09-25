import json
from eegunity.unifieddataset import UnifiedDataset

# obtain base config from file
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']
CI_output_path = config['CI_output_path']
# Test ICA function
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name, 
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.ica(output_path=CI_output_path)

print("Successfully completed ICA Test")
