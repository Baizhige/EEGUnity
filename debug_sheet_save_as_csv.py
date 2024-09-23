from eegunity.unifieddataset import UnifiedDataset
import traceback

fail_list = ['ieeedata_nju_aad','tuh_eeg_epilepsy', 'bcic_iv_4', 'physionet_sleepdephemocog', 'bcic_iv_2a', 'physionet_mssvepdb', 'physionet_sleepedfx', 'other_ahajournals', 'physionet_eegmat', 'other_artifact_rejection', 'tuh_eeg_abnormal', 'physionet_auditoryeeg', 'tuh_eeg_seizure', 'bcic_iv_2b', 'physionet_motionartifact', 'physionet_capslpdb', 'figshare_shudb', 'tuh_eeg_slowing', 'openneuro_ds003516', 'mendeley_sin', 'kaggle_inria', 'other_migrainedb', 'zenodo_3618205', 'zenodo_sin', 'iscslp2024_chineseaad', 'zenodo_saa', 'openneuro_ds004015', 'physionet_hmcsleepstaging', 'zenodo_uhd', 'zenodo_4518754', 'other_openbmi', 'physionet_eegmmidb', 'figshare_largemi', 'figshare_stroke', 'tuh_eeg', 'tuh_eeg_events', 'zenodo_7778289', 'physionet_chbmit', 'bcic_iv_1', 'other_highgammadataset', 'zenodo_kul', 'other_seed', 'other_eegdenoisenet', 'zenodo_dtu', 'github_inabiyouni', 'bcic_iv_3', 'osf_8jpc5', 'bcic_iii_2', 'figshare_meng2019', 'tuh_eeg_artifact', 'physionet_ucddb', 'kaggle_graspandlift', 'bcic_iii_1']
remain_list = ['ieee_icassp_competition_2024','bcic_iii_1','physionet_eegmmidb']
done_list = []

for folder_name in remain_list:
    try:
        unified_dataset = UnifiedDataset(domain_tag=folder_name, locator_path=f"./locator/{folder_name}_events.csv",
                                         is_unzip=False)
        unified_dataset.eeg_batch.save_as_other(output_path=f"../EEGUnity_output/{folder_name}/save_as_csv", format='csv')
    except Exception as e:
        print("fail===========")
        print(f"Folder name: {folder_name}")
        print("Error:", e)
        print("Traceback details:")
        traceback.print_exc()
