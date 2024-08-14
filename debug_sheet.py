from eegunity.unifieddataset import UnifiedDataset

unified_dataset = UnifiedDataset(domain_tag="figshare_meng2019", locator_path=r".\locator\figshare_meng2019.csv")
unified_dataset.eeg_batch.format_channel_names()
unified_dataset.eeg_correction.visualization_frequency(max_sample=16)