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
CI_output_path = config['CI_output_path']

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)
    # Test set_column function
    locator_df = pd.read_csv(f"{locator_base_path}/{folder_name}.csv")
    test_values = [1] * len(locator_df)  # Example value list matching the number of rows

    # Correct test case: setting a valid column with matching values
    try:
        unified_dataset.eeg_batch.set_column(col_name='new_column', value=test_values)
        print(f"Test for set_column with valid input for {folder_name} passed.")
    except Exception as e:
        print(f"Test for set_column with valid input for {folder_name} failed with error: {e}")

    # Test case: incorrect value list length
    wrong_length_values = [1] * (len(locator_df) - 1)  # Incorrect length
    try:
        unified_dataset.eeg_batch.set_column(col_name='new_column', value=wrong_length_values)
        print(f"Test for set_column with mismatched value length for {folder_name} failed to raise ValueError.")
    except ValueError:
        print(f"Test for set_column with mismatched value length for {folder_name} passed.")
    except Exception as e:
        print(f"Test for set_column with mismatched value length for {folder_name} failed with unexpected error: {e}")

    # Test case: invalid type for column name
    try:
        unified_dataset.eeg_batch.set_column(col_name=123, value=test_values)
        print(f"Test for set_column with invalid col_name type for {folder_name} failed to raise TypeError.")
    except TypeError:
        print(f"Test for set_column with invalid col_name type for {folder_name} passed.")
    except Exception as e:
        print(f"Test for set_column with invalid col_name type for {folder_name} failed with unexpected error: {e}")

    # Test case: invalid type for value (not a list)
    try:
        unified_dataset.eeg_batch.set_column(col_name='new_column', value='not_a_list')
        print(f"Test for set_column with invalid value type for {folder_name} failed to raise TypeError.")
    except TypeError:
        print(f"Test for set_column with invalid value type for {folder_name} passed.")
    except Exception as e:
        print(f"Test for set_column with invalid value type for {folder_name} failed with unexpected error: {e}")

print("Successfully completed all tests.")
