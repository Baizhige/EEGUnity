from eegunity.module_eeg_llm_booster.eeg_llm_des_parser import llm_description_file_parser
from eegunity.module_eeg_llm_booster.eeg_llm_file_parser import llm_boost_parser
from eegunity.share_attributes import UDatasetSharedAttributes


class EEGLLMBooster(UDatasetSharedAttributes):
    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr
        self.llm_description_file_parser = llm_description_file_parser
        self.llm_boost_parser = llm_boost_parser
