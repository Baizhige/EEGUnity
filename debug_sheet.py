from eegunity.unifieddataset import UnifiedDataset
import os
# read locator
# 遍历 raweeg 目录下的所有子文件夹
root_path = "../raweeg"  # raweeg 的父目录路径
remain_list = ['tuh_eeg_epilepsy', 'unknown_CockPartDataset', 'bcic_iv_4', 'physionet_sleepdephemocog', 'bcic_iv_2a', 'physionet_mssvepdb', 'physionet_sleepedfx', 'other_ahajournals', 'physionet_eegmat', 'other_artifact_rejection', 'tuh_eeg_abnormal', 'physionet_auditoryeeg', 'tuh_eeg_seizure', 'bcic_iv_2b', 'physionet_motionartifact']
fail_list = ['ieeedata_nju_aad']
done_list = ['physionet_capslpdb', 'figshare_shudb', 'tuh_eeg_slowing', 'openneuro_ds003516', 'mendeley_sin', 'kaggle_inria', 'other_migrainedb', 'zenodo_3618205', 'zenodo_sin', 'ieee_icassp_competition_2024', 'iscslp2024_chineseaad', 'zenodo_saa', 'openneuro_ds004015', 'physionet_hmcsleepstaging', 'zenodo_uhd', 'zenodo_4518754', 'other_openbmi', 'physionet_eegmmidb', 'figshare_largemi', 'figshare_stroke', 'tuh_eeg', 'tuh_eeg_events', 'zenodo_7778289', 'physionet_chbmit', 'bcic_iv_1', 'other_highgammadataset', 'zenodo_kul', 'other_seed', 'other_eegdenoisenet', 'zenodo_dtu', 'github_inabiyouni', 'bcic_iv_3', 'osf_8jpc5', 'bcic_iii_2', 'figshare_meng2019', 'tuh_eeg_artifact', 'physionet_ucddb', 'kaggle_graspandlift', 'bcic_iii_1']
for folder_name in remain_list:
    folder_path = os.path.join(root_path, folder_name)

    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        unified_dataset = UnifiedDataset(domain_tag=folder_name, dataset_path=folder_path, is_unzip=False)
        locator_save_path = f"./locator/{folder_name}.csv"
        unified_dataset.save_locator(locator_save_path)
#
# # second process
# unified_dataset = UnifiedDataset(domain_tag="physionet_capslpdb", locator_path="./locator/physionet_capslpdb.csv")
# unified_dataset.eeg_batch.get_events()
# unified_dataset.save_locator("./locator/physionet_capslpdb_events.csv")
