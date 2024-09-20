import copy

from eegunity.module_eeg_batch.eeg_batch import EEGBatch
from eegunity.module_eeg_correction.eeg_correction import EEGCorrection
from eegunity.module_eeg_parser.eeg_parser import EEGParser
from eegunity.share_attributes import UDatasetSharedAttributes


class UnifiedDataset(UDatasetSharedAttributes):
    def __init__(self, domain_tag, dataset_path=None, locator_path=None, is_unzip=True, verbose='CRITICAL'):
        super().__init__()
        self.set_shared_attr({'dataset_path': dataset_path})
        self.set_shared_attr({'locator_path': locator_path})
        self.set_shared_attr({'is_unzip': is_unzip})
        self.set_shared_attr({'domain_tag': domain_tag})
        self.set_shared_attr({'verbose': verbose})
        self.eeg_parser = EEGParser(self)
        self.eeg_batch = EEGBatch(self)
        self.eeg_correction = EEGCorrection(self)
        # self.module_eeg_llm_booster = LLMBooster()

    def copy(self):
        return copy.deepcopy(self)

    def save_locator(self, path):
        self.get_shared_attr()['locator'].to_csv(path, index=False)

    def get_locator(self):
        return self.get_shared_attr()['locator']

    def set_locator(self, new_locator):
        self.get_shared_attr()['locator'] = new_locator
