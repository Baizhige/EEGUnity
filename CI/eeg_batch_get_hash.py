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
CI_output_path = config['CI_output_path']+"/batch_hash"

# Test ICA function
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name, 
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    unified_dataset.eeg_batch.get_file_hashes()
    unified_dataset.save_locator(CI_output_path+'/file_hashes.csv')

print("Successfully completed hash Test")
