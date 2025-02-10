from eegunity import UnifiedDataset

u_dataset = UnifiedDataset(dataset_path='/gpfs/work/int/chengxuanqin21/science_works/EEGUnity_CI/bcic_iv_2a/', domain_tag='bcic_iv_2a')

u_dataset.save_locator('bcic_iv_2a.csv')


u_dataset2 = UnifiedDataset(dataset_path='/gpfs/work/int/chengxuanqin21/science_works/EEGUnity_CI/physionet_hmcsleepstaging/', domain_tag='physionet-hmcsleepstaging')

u_dataset2.save_locator('physionet_hmcsleepstaging.csv')
