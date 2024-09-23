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

    # Resample with specified sampling rate (downsample to 100 Hz)
    unified_dataset.eeg_batch.resample(output_path=CI_output_path, sfreq=100.0)
    print(f"Test with resample to 100 Hz for {folder_name} completed.")

    # Resample with specified sampling rate (upsample to 512 Hz)
    unified_dataset.eeg_batch.resample(output_path=CI_output_path, sfreq=512.0)
    print(f"Test with resample to 512 Hz for {folder_name} completed.")

    # Testing error handling with miss_bad_data=True
    unified_dataset.eeg_batch.resample(output_path=CI_output_path, miss_bad_data=True)
    print(f"Test with miss_bad_data=True for {folder_name} completed.")

    # Resample with additional kwargs for advanced control (for example, npad='auto')
    unified_dataset.eeg_batch.resample(output_path=CI_output_path, npad='auto')
    print(f"Test with resample and npad='auto' for {folder_name} completed.")

    # Resample with invalid parameter (to test miss_bad_data functionality)
    try:
        unified_dataset.eeg_batch.resample(output_path=CI_output_path, sfreq=-1)
    except Exception as e:
        print(f"Handled error during resample with invalid sfreq (-1) for {folder_name}: {str(e)}")

print("Successfully completed all resample tests.")
