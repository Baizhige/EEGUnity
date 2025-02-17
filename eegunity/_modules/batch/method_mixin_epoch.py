import json
import os
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import pickle
from eegunity._modules.parser.eeg_parser import get_data_row, extract_events
from eegunity.utils.h5 import h5Dataset
from eegunity.utils.handle_errors import handle_errors
from eegunity.utils.log_processing import log_processing
class EEGBatchMixinEpoch:
    def epoch_by_event(self, output_path: str,
                       exclude_bad: bool = True, miss_bad_data: bool = False,
                       get_data_row_params: Dict = None,
                       resample_params: Dict = None,
                       epoch_params: Dict = None) -> None:
        """
        Batch process EEG data to create epochs based on events specified in the event_id column. The output is saved in
        multiple .npy files for clarity and ease of use. This function serves as one of the available interfaces for
        epoch processing. Given the existence of multiple interfaces for handling epochs, we recommend using the
        unified processing interface designed specifically for this purpose. For more details, please refer to the
        documentation for UnifiedDataset.EEGBatch.process_epochs().

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Uses simple heuristics to determine bad epochs. Default is `True`.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is `False`.
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        resample_params : dict, optional
            Parameters for resampling the raw data. Must include `sfreq` for the target sampling frequency.
            Example: `{'sfreq': 256, 'npad': 'auto'}`.
        epoch_params : dict, optional
            Additional parameters for `mne.Epochs`, excluding `raw_data`, `events`, and `event_id`.

        Returns
        -------
        None
            The function modifies the dataset in place by creating and saving the epochs.

        Raises
        ------
        ValueError
            If any parameters are inconsistent or if the specified output path is invalid.
        """

        # Set default empty dictionaries if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}
        if resample_params is None:
            resample_params = {}
        if epoch_params is None:
            epoch_params = {}

        # Convert output_path to an absolute path
        output_path = os.path.abspath(output_path)

        # Initialize index.csv
        index_path = os.path.join(output_path, "index.csv")
        index_data = []

        @handle_errors(miss_bad_data)
        def app_func(row, output_path: str,
                     exclude_bad: bool = True,
                     get_data_row_params: Dict = None,
                     resample_params: Dict = None,
                     epoch_params: Dict = None):
            """Apply function to process individual rows."""
            print(f"Processing File: {row['File Path']}")
            # Load raw data with additional parameters
            raw_data = get_data_row(row, **get_data_row_params)

            # Resample if resample_params includes `sfreq`
            if 'sfreq' in resample_params:
                raw_data.resample(**resample_params)

            # Extract events
            events, event_id = extract_events(raw_data)
            # Create epochs with the passed epoch_params
            epochs = mne.Epochs(raw_data, events, event_id, **epoch_params)

            # Exclude bad epochs
            if exclude_bad:
                epochs.drop_bad()

            # Iterate through each event type
            for event in event_id:
                event_epochs = epochs[event]
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                event_output_path = os.path.join(output_path, event)
                os.makedirs(event_output_path, exist_ok=True)

                # Construct absolute file path for saving
                file_path = os.path.abspath(
                    os.path.join(event_output_path, f"{file_name}_{event}_epoch.npy")
                )
                # Validate data export
                epoch_data = event_epochs.get_data()
                if epoch_data.ndim != 3:
                    raise ValueError("Epoch data is not three-dimensional.")

                # Save data
                np.save(file_path, epoch_data)
                print(f"Epoch File was saved to {file_path}")

                # Gather metadata for index.csv
                description = json.loads(epochs.info['description'])
                eegunity_desc = description['eegunity_description']
                index_data.append({
                    "File Path": file_path,
                    "Class Name": event,
                    "Number of Epoch": len(event_epochs),
                    "Channel Names": ",".join(event_epochs.info['ch_names']),
                    "Amplifer": eegunity_desc.get('amplifier', 'unknown'),
                    "Cap": eegunity_desc.get('cap', 'unknown'),
                    "Age": eegunity_desc.get('age', 'unknown'),
                    "Gender": eegunity_desc.get('sex', 'unknown'),
                    "Handeness": eegunity_desc.get('handedness', 'unknown'),
                    "Comment": ""
                })

        # Use batch_process to process data
        self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: app_func(
                row,
                output_path,
                exclude_bad=exclude_bad,
                get_data_row_params=get_data_row_params,
                resample_params=resample_params,
                epoch_params=epoch_params
            ),
            is_patch=False,
            result_type=None
        )

        # Save index.csv
        pd.DataFrame(index_data).to_csv(index_path, index=False)
        print(f"Index file saved to {index_path}")


    def epoch_by_long_event(self, output_path: str,
                            overlap: float,
                            exclude_bad: bool = True,
                            miss_bad_data: bool = False,
                            get_data_row_params: Dict = None,
                            resample_params: Dict = None,
                            epoch_params: Dict = None) -> None:
        """
        Batch process EEG data to create epochs for long-duration events with overlapping segments.
        This function serves as one of the available interfaces for epoch processing. Given the existence of multiple
        interfaces for handling epochs, we recommend using the unified processing interface designed specifically for
        this purpose. For more details, please refer to the documentation for UnifiedDataset.EEGBatch.process_epochs().

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        overlap : float
            Proportion of overlap between consecutive segments (0.0 to 1.0).
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Uses simple heuristics to determine bad epochs. Default is `True`.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is `False`.
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        resample_params : dict, optional
            Parameters for resampling the raw data. Must include `sfreq` for the target sampling frequency.
            Example: `{'sfreq': 256, 'npad': 'auto'}`.
        epoch_params : dict, optional
            Additional parameters for `mne.Epochs`, excluding `raw_data`, `events`, and `event_id`.

        Returns
        -------
        None
            The function modifies the dataset in place by creating and saving the epochs.

        Raises
        ------
        ValueError
            If any parameters are inconsistent or if the specified output path is invalid.
        """
        # Set default empty dictionaries if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}
        if resample_params is None:
            resample_params = {}
        if epoch_params is None:
            epoch_params = {}

        if not 0.0 <= overlap < 1.0:
            raise ValueError("Overlap must be between 0.0 (no overlap) and less than 1.0.")

        # Convert output_path to an absolute path
        output_path = os.path.abspath(output_path)

        # Initialize index.csv
        index_path = os.path.join(output_path, "index.csv")
        index_data = []

        @handle_errors(miss_bad_data)
        def app_func(row, output_path: str,
                     overlap: float,
                     exclude_bad: bool = True,
                     get_data_row_params: Dict = None,
                     resample_params: Dict = None,
                     epoch_params: Dict = None):
            """Apply function to process individual rows."""
            print(f"Processing File: {row['File Path']}")

            # Load raw data with additional parameters
            raw_data = get_data_row(row, **get_data_row_params)

            # Resample if resample_params includes `sfreq`
            if 'sfreq' in resample_params:
                raw_data.resample(**resample_params)

            # Check for annotations in the raw data
            annotations = raw_data.annotations
            if not annotations:
                raise ValueError(f"No annotations found in the raw data for {row['File Path']}.")

            # Extract epoch length from epoch_params
            t_min = epoch_params.get('tmin', 0)
            t_max = epoch_params.get('tmax', 1)
            epoch_length = t_max - t_min  # Duration in seconds

            # Create temporary annotations for segmentation of long events
            temp_annotations = []
            for onset, duration, description in zip(annotations.onset, annotations.duration,
                                                    annotations.description):
                if duration > epoch_length:
                    # Calculate the step size based on overlap
                    step = epoch_length * (1 - overlap)
                    num_segments = int((duration - epoch_length) // step) + 1
                    for i in range(num_segments):
                        segment_start = onset + i * step
                        temp_annotations.append((segment_start, epoch_length, description))

            if not temp_annotations:
                raise ValueError(
                    f"For long events (such as sleep stages), the duration must be greater than (t_max - t_min). "
                    f"Current duration is {duration}s, while the segmentation length is {t_max - t_min}s. "
                    "Please check the event duration or the 'epoch_param' settings."
                )

            # Convert the temporary annotations to mne.Annotations
            long_event_annotations = mne.Annotations(
                onset=[a[0] for a in temp_annotations],
                duration=[a[1] for a in temp_annotations],
                description=[a[2] for a in temp_annotations]
            )
            raw_data.set_annotations(long_event_annotations)

            # Create epochs based on the temporary annotations
            events, event_id_map = mne.events_from_annotations(raw_data)
            epochs = mne.Epochs(raw_data, events, event_id_map, **epoch_params)

            # Exclude bad epochs
            if exclude_bad:
                epochs.drop_bad()

            available_descriptions = list(epochs.event_id.keys())

            # Save each segmented epoch
            for description in available_descriptions:
                event_epochs = epochs[description]
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                event_output_path = os.path.join(output_path, description)
                os.makedirs(event_output_path, exist_ok=True)

                file_path = os.path.abspath(
                    os.path.join(event_output_path, f"{file_name}_{description}_epoch.npy")
                )
                try:
                    # Validate data export
                    epoch_data = event_epochs.get_data()
                    if epoch_data.ndim != 3:
                        raise ValueError("Epoch data is not three-dimensional.")

                    # Save data
                    np.save(file_path, epoch_data)
                    print(f"Epoch File was saved to {file_path}")

                    # Gather metadata for index.csv
                    index_data.append({
                        "File Path": file_path,
                        "Class Name": description,
                        "Number of Epoch": len(event_epochs),
                        "Channel Names": ",".join(event_epochs.info['ch_names']),
                        "Comment": ""
                    })
                except Exception as e:
                    index_data.append({
                        "File Path": file_path,
                        "Class Name": description,
                        "Number of Epoch": "",
                        "Channel Names": "",
                        "Comment": str(e)
                    })

        # Use batch_process to process data
        self.batch_process(
            lambda row: True,
            lambda row: app_func(
                row,
                output_path,
                overlap,
                exclude_bad=exclude_bad,
                get_data_row_params=get_data_row_params,
                resample_params=resample_params,
                epoch_params=epoch_params
            ),
            is_patch=False,
            result_type=None
        )

        # Save index.csv
        pd.DataFrame(index_data).to_csv(index_path, index=False)
        print(f"Index file saved to {index_path}")


    def epoch_by_event_hdf5(self, output_path: str,
                            exclude_bad: bool = True,
                            file_name_prefix: str = "EpochData",
                            miss_bad_data: bool = False,
                            get_data_row_params: Dict = None,
                            resample_params: Dict = None,
                            epoch_params: Dict = None) -> None:
        """
        Batch process EEG data to create epochs based on events specified in the event_id column,
        save the results in HDF5 format, and generate a JSON file with event counts.
        This function serves as one of the available interfaces for epoch processing. Given the existence of multiple
        interfaces for handling epochs, we recommend using the unified processing interface designed specifically for
        this purpose. For more details, please refer to the documentation for UnifiedDataset.EEGBatch.process_epochs().

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Uses simple heuristics to determine bad epochs. Default is `True`.
        file_name_prefix : str, optional
            The filename prefix to save hdf5 and event info file. Default is `EpochData`.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is `False`.
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        resample_params : dict, optional
            Parameters for resampling the raw data. Must include `sfreq` for the target sampling frequency.
            Example: `{'sfreq': 256, 'npad': 'auto'}`.
        epoch_params : dict, optional
            Additional parameters for `mne.Epochs`, excluding `raw_data`, `events`, and `event_id`.

        Returns
        -------
        None
            The function modifies the dataset in place by creating and saving the epochs in HDF5 format,
            and generates an event_info.json file.
        """
        # Set default empty dictionaries if parameters are None
        if get_data_row_params is None:
            get_data_row_params = {}
        if resample_params is None:
            resample_params = {}
        if epoch_params is None:
            epoch_params = {}

        # Ensure output_path exists
        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)

        # Initialize HDF5 dataset
        dataset = h5Dataset(Path(output_path), name=file_name_prefix)

        # Dictionary to keep track of event counts
        event_info = {}

        @handle_errors(miss_bad_data)
        @log_processing
        def app_func(row, exclude_bad: bool = True,
                     get_data_row_params: Dict = None,
                     resample_params: Dict = None,
                     epoch_params: Dict = None):
            """
            Process an individual file to create and save epochs.
            """
            raw_data = get_data_row(row, **get_data_row_params)

            # Resample raw data if necessary
            if 'sfreq' in resample_params:
                raw_data.resample(**resample_params)

            # Extract events
            events, event_id = extract_events(raw_data)

            # Create epochs
            epochs = mne.Epochs(raw_data, events, event_id, **epoch_params)

            # Exclude bad epochs
            if exclude_bad:
                epochs.drop_bad()

            # Create a group for the current file
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            grp = dataset.addGroup(grpName=file_name)
            # Add info dataset for the group
            info_bytes = pickle.dumps(raw_data.info)
            info_array = np.frombuffer(info_bytes, dtype='uint8')
            dataset.addDataset(grp, 'info', info_array, chunks=None)
            # Iterate through each event type and save data
            for event in event_id:
                try:
                    event_epochs = epochs[event]
                    epoch_data = event_epochs.get_data()

                    if epoch_data.ndim != 3:
                        raise ValueError("Epoch data is not three-dimensional.")

                    # Add dataset for this event
                    dset = dataset.addDataset(grp, event, epoch_data, chunks=epoch_data.shape)

                    # Add attributes for the dataset
                    dataset.addAttributes(dset, 'rsFreq', raw_data.info['sfreq'])
                    dataset.addAttributes(dset, 'chOrder', event_epochs.info['ch_names'])

                    # Update event counts
                    event_info[event] = event_info.get(event, 0) + len(event_epochs)

                    print(f"Processed and saved epochs for event: {event}")

                except Exception as e:
                    # Handle error and skip this event
                    print(f"Error processing event '{event}': {e}")
                    continue

            print(f"Processed and saved epochs for file: {file_name}")

        # Use batch_process to process data
        self.batch_process(
            lambda row: True,
            lambda row: app_func(
                row,
                exclude_bad=exclude_bad,
                get_data_row_params=get_data_row_params,
                resample_params=resample_params,
                epoch_params=epoch_params
            ),
            is_patch=False,
            result_type=None
        )

        # Save the HDF5 dataset
        dataset.save()
        print(f"HDF5 dataset saved to {output_path}")

        # Save event_info.json
        event_info_path = os.path.join(output_path, file_name_prefix + "_event_info.json")
        with open(event_info_path, 'w') as f:
            json.dump(event_info, f, indent=4)
        print(f"Event information saved to {event_info_path}")


    def epoch_by_long_event_hdf5(self, output_path: str,
                                 overlap: float,
                                 file_name_prefix: str = "EpochData",
                                 exclude_bad: bool = True,
                                 miss_bad_data: bool = False,
                                 get_data_row_params: Dict = None,
                                 resample_params: Dict = None,
                                 epoch_params: Dict = None) -> None:
        """
        Batch process EEG data to create epochs for long-duration events with overlapping segments
        and save the results in HDF5 format. Also generates event_info.json containing event counts.
        This function serves as one of the available interfaces for epoch processing. Given the existence of multiple
        interfaces for handling epochs, we recommend using the unified processing interface designed specifically for
        this purpose. For more details, please refer to the documentation for UnifiedDataset.EEGBatch.process_epochs().

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        overlap : float
            Proportion of overlap between consecutive segments (0.0 to 1.0).
        file_name_prefix : str, optional
            The filename prefix to save hdf5 and event info file. Default is `EpochData`.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Default is `True`.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is `False`.
        get_data_row_params : dict, optional
            Additional parameters passed to `get_data_row()` for data retrieval.
        resample_params : dict, optional
            Parameters for resampling the raw data.
        epoch_params : dict, optional
            Additional parameters for `mne.Epochs`.

        Returns
        -------
        None
            The function creates and saves the epochs in HDF5 format and event_info.json.

        Raises
        ------
        ValueError
            If any parameters are inconsistent.
        """
        if get_data_row_params is None:
            get_data_row_params = {}
        if resample_params is None:
            resample_params = {}
        if epoch_params is None:
            epoch_params = {}

        if not 0.0 <= overlap < 1.0:
            raise ValueError("Overlap must be between 0.0 (no overlap) and less than 1.0.")

        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)

        # Initialize the HDF5 dataset
        dataset = h5Dataset(Path(output_path), name=file_name_prefix)

        # Initialize event count dictionary
        event_info = {}

        @handle_errors(miss_bad_data)
        def process_file(row):
            """
            Process an individual file to create epochs from long events.
            """
            print(f"Processing File: {row['File Path']}")
            raw_data = get_data_row(row, **get_data_row_params)

            # Resample the raw data if necessary
            if 'sfreq' in resample_params:
                raw_data.resample(**resample_params)

            # Check for annotations
            annotations = raw_data.annotations
            if not annotations:
                raise ValueError(f"No annotations found in the raw data for {row['File Path']}.")

            # Calculate epoch length and create overlapping segments
            t_min = epoch_params.get('tmin', 0)
            t_max = epoch_params.get('tmax', 1)
            epoch_length = t_max - t_min

            temp_annotations = []
            for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
                if duration > epoch_length:
                    step = epoch_length * (1 - overlap)
                    num_segments = int((duration - epoch_length) // step) + 1
                    for i in range(num_segments):
                        segment_start = onset + i * step
                        temp_annotations.append((segment_start, epoch_length, description))

            if not temp_annotations:
                raise ValueError("No long events long enough for segmentation.")

            # Convert the temporary annotations to mne.Annotations
            long_event_annotations = mne.Annotations(
                onset=[a[0] for a in temp_annotations],
                duration=[a[1] for a in temp_annotations],
                description=[a[2] for a in temp_annotations]
            )
            raw_data.set_annotations(long_event_annotations)

            # Create epochs
            events, event_id_map = mne.events_from_annotations(raw_data)
            epochs = mne.Epochs(raw_data, events, event_id_map, **epoch_params)

            if exclude_bad:
                epochs.drop_bad()

            # Create a group for the current file in HDF5
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            grp = dataset.addGroup(grpName=file_name)
            # Save info metadata
            info_bytes = pickle.dumps(raw_data.info)
            info_array = np.frombuffer(info_bytes, dtype='uint8')
            dataset.addDataset(grp, 'info', info_array, chunks=None)
            # Save epochs to HDF5
            for description, event_id in event_id_map.items():
                try:
                    event_epochs = epochs[description]
                    epoch_data = event_epochs.get_data()

                    if epoch_data.ndim != 3:
                        raise ValueError("Epoch data is not three-dimensional.")

                    # Add dataset for this event
                    dset = dataset.addDataset(grp, description, epoch_data, chunks=epoch_data.shape)

                    # Add metadata
                    dataset.addAttributes(dset, 'rsFreq', raw_data.info['sfreq'])
                    dataset.addAttributes(dset, 'chOrder', event_epochs.info['ch_names'])

                    # Update event_info
                    event_info[description] = event_info.get(description, 0) + len(event_epochs)
                except Exception as e:
                    # Handle error and skip this event
                    print(f"Error processing event '{description}': {e}")
                    continue

            print(f"Finished processing {file_name}.")

        # Use batch_process to process files
        self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: process_file(row),
            is_patch=False,
            result_type=None
        )

        # Save the HDF5 dataset
        dataset.save()
        print(f"Long-event epochs saved to {output_path}.")

        # Save event_info.json
        event_info_path = os.path.join(output_path, file_name_prefix + "_event_info.json")
        with open(event_info_path, 'w') as f:
            json.dump(event_info, f, indent=4)
        print(f"Event information saved to {event_info_path}.")


    def process_epochs(self,
                       output_path: str,
                       long_event: bool = False,
                       overlap: float = 0,
                       use_hdf5: bool = True,
                       file_name_prefix: str = "EpochData",
                       exclude_bad: bool = True,
                       miss_bad_data: bool = False,
                       get_data_row_params: dict = None,
                       resample_params: dict = None,
                       epoch_params: dict = None) -> None:
        """
        Unified interface for processing epochs.

        This method selects and calls one of the underlying epoch processing methods based
        on the provided parameters:
          - If long_event is False and use_hdf5 is False, it calls epoch_by_event.
          - If long_event is False and use_hdf5 is True, it calls epoch_by_event_hdf5.
          - If long_event is True and use_hdf5 is False, it calls epoch_by_long_event.
          - If long_event is True and use_hdf5 is True, it calls epoch_by_long_event_hdf5.

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        long_event : bool, optional
            Whether to process long-duration events. If True, overlap must be provided. Default is False.
        overlap : float, optional
            Overlap ratio for long events (0.0 <= overlap < 1.0). Required if long_event is True.
            Default is 0 (non-overlap).
        use_hdf5 : bool, optional
            Whether to save the results in HDF5 format. If you are working with deep learning, especially large models,
            we strongly recommend using this interface (use_hdf5=True) for faster processing. Default is True.
        file_name_prefix : str, optional
            Filename prefix for HDF5 saving (used only if use_hdf5 is True).  Default is 'EpochData'.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Default is True.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is True.
        get_data_row_params : dict, optional
            Additional parameters for data retrieval get_data_row().
        resample_params : dict, optional
            Parameters for resampling the raw data, mne.io.raw.resample()
        epoch_params : dict, optional
            Additional parameters for creating epochs.

        Returns
        -------
        None
            The method processes and saves the epochs by calling the appropriate underlying method.
        """
        if long_event and overlap is None:
            raise ValueError("Overlap must be provided when long_event is True.")

        # Determine positional arguments: include 'overlap' only for long events.
        args = [output_path] if not long_event else [output_path, overlap]

        # Build common keyword arguments, adding file_name_prefix if HDF5 is used.
        kwargs = {
            "exclude_bad": exclude_bad,
            "miss_bad_data": miss_bad_data,
            "get_data_row_params": get_data_row_params,
            "resample_params": resample_params,
            "epoch_params": epoch_params,
            **({"file_name_prefix": file_name_prefix} if use_hdf5 else {})
        }

        # Mapping from (long_event, use_hdf5) to the corresponding method.
        method_mapping = {
            (False, False): self.epoch_by_event,
            (False, True): self.epoch_by_event_hdf5,
            (True, False): self.epoch_by_long_event,
            (True, True): self.epoch_by_long_event_hdf5
        }

        # Call the selected method with the prepared arguments.
        method_mapping[(long_event, use_hdf5)](*args, **kwargs)