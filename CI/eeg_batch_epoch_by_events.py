import json
import os
import sys

# 获取当前脚本的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将父目录添加到 sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from eegunity.unifieddataset import UnifiedDataset

# 从配置文件读取基本配置信息
with open('CI_config.json', 'r') as config_file:
    config = json.load(config_file)

remain_list = config['test_data_list']
locator_base_path = config['locator_base_path']
CI_output_path = config['CI_output_path']

# 依次测试 process_epochs 方法的四种情况
for folder_name in remain_list:
    unified_dataset = UnifiedDataset(domain_tag=folder_name,
                                     locator_path=f"{locator_base_path}/{folder_name}.csv",
                                     is_unzip=False)

    # # Case 1: long_event=False, use_hdf5=False → 调用 epoch_by_event
    # unified_dataset.eeg_batch.process_epochs(output_path=CI_output_path+"/epoch_by_events", long_event=False, use_hdf5=False,
    #                                           epoch_params={'tmin': 0,
    #                                           'tmax': 4,
    #                                           'baseline': None,
    #                                           'event_repeated': 'merge'})
    # print(f"Test case 1 (long_event=False, use_hdf5=False) for {folder_name} completed.")
    #
    # # Case 2: long_event=False, use_hdf5=True → 调用 epoch_by_event_hdf5
    # unified_dataset.eeg_batch.process_epochs(output_path=CI_output_path+"/epoch_by_events_hdf5", long_event=False, use_hdf5=True,
    #                                          epoch_params={'tmin': 0,
    #                                           'tmax': 4,
    #                                           'baseline': None,
    #                                           'event_repeated': 'merge'})
    # print(f"Test case 2 (long_event=False, use_hdf5=True) for {folder_name} completed.")
    #
    # # Case 3: long_event=True, use_hdf5=False → 调用 epoch_by_long_event
    # unified_dataset.eeg_batch.process_epochs(output_path=CI_output_path+"/epoch_by_long_events", long_event=True, use_hdf5=False, overlap=0.5,
    #                                          epoch_params={'tmin': 0,
    #                                           'tmax': 4,
    #                                           'baseline': None,
    #                                           'event_repeated': 'merge'})
    # print(f"Test case 3 (long_event=True, use_hdf5=False) for {folder_name} completed.")

    # Case 4: long_event=True, use_hdf5=True → 调用 epoch_by_long_event_hdf5
    unified_dataset.eeg_batch.process_epochs(output_path=CI_output_path+"/epoch_by_long_events_hdf5", long_event=True, use_hdf5=True, overlap=0.5,
                                             epoch_params={'tmin': 0,
                                              'tmax': 4,
                                              'baseline': None,
                                              'event_repeated': 'merge'})
    print(f"Test case 4 (long_event=True, use_hdf5=True) for {folder_name} completed.")

print("Successfully completed all process_epochs tests.")
