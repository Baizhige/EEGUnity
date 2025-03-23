import copy
from eegunity.modules.batch.eeg_batch import EEGBatch
from eegunity.modules.correction.eeg_correction import EEGCorrection
from eegunity.modules.parser.eeg_parser import EEGParser
from eegunity.modules.llm_booster.eeg_llm_booster import EEGLLMBooster
from eegunity._share_attributes import _UDatasetSharedAttributes


class UnifiedDataset(_UDatasetSharedAttributes):
    """
        This is the kernel class to manage mutiple EEG datasets and associated processing tools.

        Attributes:
        -----------
        dataset_path : str, optional
            Path to the dataset (folder). Should not be provided alongside locator_path.
        locator_path : str, optional
            Path to the locator. Should not be provided alongside dataset_path.
        is_unzip : bool, optional
            If set to True, any Zip files in the specified dataset will be unzipped. Be aware that unzipping may modify the dataset.
        domain_tag : str, optional
            The domain tag identifies the dataset name and is required if you specify a dataset path.
        verbose : str, optional
            Level of verbosity for logging (default is 'CRITICAL').
        eeg_parser : EEGParser
            EEGParser module
        eeg_batch : EEGBatch
            EEGBatch module
        eeg_correction : EEGCorrection
            EEGCorrection module
        llm_booster : EEGLLMBooster
           EEGLLMBooster module
        """

    def __init__(self, dataset_path: str = None, locator_path: str = None, domain_tag: str = None,
                 is_unzip: bool = True, verbose: str = 'CRITICAL'):
        """
        Initialize the class with either dataset_path or locator_path. Only one of
        these parameters should be provided. If dataset_path is provided, domain_tag is required.

        Parameters:
        -----------
        dataset_path : str, optional
            Path to the dataset (folder). Note: Do not provide dataset_path if you are using locator_path.
        locator_path : str, optional
            The file path to the locator (a CSV-like file) that stores all metadata for the UnifiedDataset in EEGUnity. Note: Do not provide locator_path if you are using dataset_path.
        domain_tag : str, optional
            The domain tag identifies the dataset name.  Note: Do not provide domain_tag if you are using locator_path.
        is_unzip : bool, optional
            A flag indicating whether the dataset should be unzipped (default is True).
        verbose : str, optional
            The verbosity level for logging (default is 'CRITICAL').

        Raises:
        -------
        ValueError
            If both dataset_path and locator_path are provided, or neither is provided.
        Examples
        --------
        >>> unified_dataset = UnifiedDataset("path/to/your/dataset")
        >>> unified_dataset_locator = UnifiedDataset(locator_path="path/to/your/locator.csv")
        """
        super().__init__()

        # Ensure only one of dataset_path or locator_path is provided
        if dataset_path and locator_path:
            raise ValueError("Only one of 'dataset_path' or 'locator_path' can be provided, not both.")
        if not dataset_path and not locator_path:
            raise ValueError("One of 'dataset_path' or 'locator_path' must be provided.")

        # Ensure domain_tag is provided when dataset_path is used
        if dataset_path and not domain_tag:
            domain_tag = "not_specified"

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
        """
        Create a deep copy of the UnifiedDataset instance.

        Returns:
        --------
        UnifiedDataset
            A deep copy of the current UnifiedDataset instance.
        """
        return copy.deepcopy(self)

    def save_locator(self, path):
        """
        Save the locator of this UnifiedDataset to a CSV file at the specified path. This file is helpful for checking the current status and metadata after data processing.
        You can also reload the UnifiedDataset later by using this locator file, for example:
        unified_dataset = UnifiedDataset(locator_path="your_locator_path")

        Parameters:
        -----------
        path : str
            The file path where the locator should be saved.
        """
        self.get_shared_attr()['locator'].to_csv(path, index=False)

    def get_locator(self):
        """
        Return the locator in DataFrame.

        Returns:
        --------
        pandas.DataFrame
            The locator DataFrame associated with the dataset.
        """
        return self.get_shared_attr()['locator']

    def set_locator(self, new_locator):
        """
        Set a new locator for this UnifiedDataset instance.
        This allows you to update the metadata for the entire dataset without altering the original raw file.

        Parameters:
        -----------
        new_locator : pandas.DataFrame
            The new locator DataFrame to associate with the dataset.
        """
        self.get_shared_attr()['locator'] = new_locator

    def group_by_domain(self):
        """
        Groups the locator data by the 'domain_tag' and returns multiple UnifiedDataset instances.

        Returns:
        --------
        List[UnifiedDataset] : A list of UnifiedDataset instances, each grouped by domain.
        """
        locator = self.get_shared_attr()['locator']

        if 'Domain Tag' not in locator.columns:
            raise ValueError("Locator dataframe must have a 'Domain Tag' column.")

        # Group the locator data by 'domain_tag'
        grouped = locator.groupby('Domain Tag')

        datasets = []
        for domain, grouped_locator in grouped:
            # Create a new UnifiedDataset instance
            new_dataset = self.copy()
            # Set the new locator for the instance
            new_dataset.set_locator(grouped_locator)
            # Add the new instance to the result list
            datasets.append(new_dataset)

        return datasets