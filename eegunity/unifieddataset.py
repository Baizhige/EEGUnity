import copy

from eegunity.module_eeg_batch.eeg_batch import EEGBatch
from eegunity.module_eeg_correction.eeg_correction import EEGCorrection
from eegunity.module_eeg_parser.eeg_parser import EEGParser
from eegunity.module_eeg_llm_booster.eeg_llm_booster import EEGLLMBooster
from eegunity.share_attributes import UDatasetSharedAttributes


class UnifiedDataset(UDatasetSharedAttributes):
    """
        This is the kernel class to manage mutiple EEG dataset and associated processing tools.

        This class allows for the initialization of EEG data using either a
        dataset path or a locator path, with the option to unzip the potential zip file.
        If a dataset path is provided, a domain tag must also be provided.

        Attributes:
        -----------
        dataset_path : str, optional
            Path to the dataset. Should not be provided alongside locator_path.
        locator_path : str, optional
            Path to the locator. Should not be provided alongside dataset_path.
        is_unzip : bool, optional
            A flag indicating if the dataset should be unzipped (default is True).
        domain_tag : str, optional
            Domain tag, required when dataset_path is provided.
        verbose : str, optional
            Level of verbosity for logging (default is 'CRITICAL').
        eeg_parser : EEGParser
            EEGParser module
        eeg_batch : EEGBatch
            EEGBatch module
        eeg_correction : EEGCorrection
            EEGCorrection module
        module_eeg_llm_booster : EEGLLMBooster
           EEGLLMBooster module
        """

    def __init__(self, domain_tag: str = None, dataset_path: str = None, locator_path: str = None,
                 is_unzip: bool = True, verbose: str = 'CRITICAL'):
        """
        Initialize the class with either dataset_path or locator_path. Only one of
        these parameters should be provided. If dataset_path is provided, domain_tag is required.

        Parameters:
        -----------
        domain_tag : str, optional
            The domain tag, required when dataset_path is provided.
        dataset_path : str, optional
            The path to the dataset. Should not be provided alongside locator_path.
        locator_path : str, optional
            The path to the locator. Should not be provided alongside dataset_path.
        is_unzip : bool, optional
            A flag indicating whether the dataset should be unzipped (default is True).
        verbose : str, optional
            The verbosity level for logging (default is 'CRITICAL').

        Raises:
        -------
        ValueError
            If both dataset_path and locator_path are provided, or neither is provided.
            If dataset_path is provided without domain_tag.
        """
        super().__init__()

        # Ensure only one of dataset_path or locator_path is provided
        if dataset_path and locator_path:
            raise ValueError("Only one of 'dataset_path' or 'locator_path' can be provided, not both.")
        if not dataset_path and not locator_path:
            raise ValueError("One of 'dataset_path' or 'locator_path' must be provided.")

        # Ensure domain_tag is provided when dataset_path is used
        if dataset_path and not domain_tag:
            raise ValueError("A 'domain_tag' must be provided when 'dataset_path' is specified.")

        # Set attributes
        self.set_shared_attr({'dataset_path': dataset_path})
        self.set_shared_attr({'locator_path': locator_path})
        self.set_shared_attr({'is_unzip': is_unzip})
        self.set_shared_attr({'domain_tag': domain_tag})
        self.set_shared_attr({'verbose': verbose})

        # Initialize associated modules
        self.eeg_parser = EEGParser(self)
        self.eeg_batch = EEGBatch(self)
        self.eeg_correction = EEGCorrection(self)
        self.module_eeg_llm_booster = EEGLLMBooster(self)

    def copy(self):
        return copy.deepcopy(self)

    def save_locator(self, path):
        self.get_shared_attr()['locator'].to_csv(path, index=False)

    def get_locator(self):
        return self.get_shared_attr()['locator']

    def set_locator(self, new_locator):
        self.get_shared_attr()['locator'] = new_locator
