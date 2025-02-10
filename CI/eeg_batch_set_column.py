import json
import pandas as pd
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
CI_output_path = config['CI_output_path']+"/set_column"

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    # Test set_column function
    locator_df = pd.read_csv(f"{locator_base_path}/{folder_name}.csv")
    test_values = [1] * len(locator_df)  # Example value list matching the number of rows

    # Correct test case: setting a valid column with matching values
    unified_dataset.eeg_batch.set_metadata(col_name='new_column', value=test_values)
    print(f"Test for set_column with valid input for {folder_name} passed.")


print("Successfully completed all tests.")
