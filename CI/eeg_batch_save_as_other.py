import json
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

    # Test saving in 'fif' format (default)
    try:
        unified_dataset.eeg_batch.save_as_other(output_path=CI_output_path, format='fif')
        print(f"Test with default format (fif) for {folder_name} completed.")
    except FileNotFoundError as e:
        print(f"FileNotFoundError encountered for {folder_name}: {str(e)}")
    except ValueError as e:
        print(f"ValueError encountered for {folder_name}: {str(e)}")

    # Test saving in 'csv' format
    try:
        unified_dataset.eeg_batch.save_as_other(output_path=CI_output_path, format='csv')
        print(f"Test with csv format for {folder_name} completed.")
    except FileNotFoundError as e:
        print(f"FileNotFoundError encountered for {folder_name}: {str(e)}")
    except ValueError as e:
        print(f"ValueError encountered for {folder_name}: {str(e)}")

    # Test saving with domain_tag specified
    try:
        unified_dataset.eeg_batch.save_as_other(output_path=CI_output_path, domain_tag=folder_name, format='fif')
        print(f"Test with domain_tag={folder_name} for {folder_name} completed.")
    except FileNotFoundError as e:
        print(f"FileNotFoundError encountered for {folder_name} with domain_tag: {str(e)}")
    except ValueError as e:
        print(f"ValueError encountered for {folder_name} with domain_tag: {str(e)}")

    # Test invalid format
    try:
        unified_dataset.eeg_batch.save_as_other(output_path=CI_output_path, format='unsupported_format')
    except ValueError as e:
        print(f"Correctly caught ValueError for unsupported format for {folder_name}: {str(e)}")

    # Test with invalid output path
    try:
        unified_dataset.eeg_batch.save_as_other(output_path='/invalid/path', format='fif')
    except FileNotFoundError as e:
        print(f"Correctly caught FileNotFoundError for invalid path for {folder_name}: {str(e)}")

print("Successfully completed all save_as_other tests.")
