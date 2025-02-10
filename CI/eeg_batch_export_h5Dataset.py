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
CI_output_path = config['CI_output_path']+"/export_h5Dataset"

# Initialize a flag to track test results
all_tests_passed = True

# Test function with different parameter combinations
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(
        domain_tag=folder_name,
        locator_path=f"{locator_base_path}/{folder_name}.csv",
        is_unzip=False
    )
    # Set output path
    output_path = f"{CI_output_path}/{folder_name}"

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Correct test case: valid output_path and name
    try:
        unified_dataset.eeg_batch.export_h5Dataset(
            output_path=output_path,
            name='EEGUnity_export2',
            verbose=False
        )
        print(f"Test for export_h5Dataset with valid input for {folder_name} passed.")
    except Exception as e:
        print(f"Test for export_h5Dataset with valid input for {folder_name} failed with error: {e}")
        all_tests_passed = False

    # Test case: output_path does not exist
    invalid_output_path = f"{CI_output_path}/non_existent_directory_{folder_name}"

    try:
        unified_dataset.eeg_batch.export_h5Dataset(
            output_path=invalid_output_path,
            name='EEGUnity_export',
            verbose=False
        )
        print(f"Test for export_h5Dataset with non-existent output_path for {folder_name} failed to raise FileNotFoundError.")
        all_tests_passed = False
    except FileNotFoundError:
        print(f"Test for export_h5Dataset with non-existent output_path for {folder_name} passed.")
    except Exception as e:
        print(f"Test for export_h5Dataset with non-existent output_path for {folder_name} failed with unexpected error: {e}")
        all_tests_passed = False

    # Test case: name is not a string
    try:
        unified_dataset.eeg_batch.export_h5Dataset(
            output_path=output_path,
            name=12345,  # Invalid name type
            verbose=False
        )
        print(f"Test for export_h5Dataset with invalid name type for {folder_name} failed to raise TypeError.")
        all_tests_passed = False
    except TypeError:
        print(f"Test for export_h5Dataset with invalid name type for {folder_name} passed.")
    except Exception as e:
        print(f"Test for export_h5Dataset with invalid name type for {folder_name} failed with unexpected error: {e}")
        all_tests_passed = False

if all_tests_passed:
    print("All tests passed successfully.")
else:
    print("CI tests failed.")
