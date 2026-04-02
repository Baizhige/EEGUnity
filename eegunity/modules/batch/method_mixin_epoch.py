import json
import os
import warnings
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional
import pickle
from eegunity.modules.parser.eeg_parser import extract_events
from eegunity.utils.h5 import h5Dataset, h5EpochDatasetV2
from eegunity.utils.handle_errors import handle_errors
from eegunity.utils.log_processing import log_processing
from eegunity.utils.label_channel import resample_raw_with_labels

_V1_DEPRECATION_MSG = (
    "format_version='v1' is deprecated and will be removed in a future release. "
    "Use format_version='v2' (default) for better IO performance, smaller file size, "
    "and PyTorch-friendly random access."
)
class EEGBatchMixinEpoch:
    def epoch_by_event(self, *args, **kwargs) -> None:
        """
        .. deprecated::
            This method is no longer maintained and will be removed in a future
            release. Use :meth:`epoch_by_event_hdf5` instead, which produces a
            single HDF5 file with significantly faster IO and smaller storage.

        Raises
        ------
        NotImplementedError
            Always. Migrate to ``epoch_by_event_hdf5``.
        """
        raise NotImplementedError(
            "epoch_by_event() (npy output) is no longer maintained. "
            "Use epoch_by_event_hdf5() instead: it produces a single HDF5 file "
            "with faster random-access IO and smaller file size. "
            "See the EEGUnity documentation for migration details."
        )

        # Convert output_path to an absolute path
        output_path = os.path.abspath(output_path)

        # Initialize index.csv
        index_path = os.path.join(output_path, "index.csv")

        @handle_errors(miss_bad_data)
        def app_func(row, output_path: str,
                     exclude_bad: bool = True,
                     get_data_row_params: Dict = None,
                     resample_params: Dict = None,
                     epoch_params: Dict = None):
            """Apply function to process individual rows."""
            print(f"Processing File: {row['File Path']}")
            # Load raw data with additional parameters
            raw_data = self._get_data_row(row, **get_data_row_params)

            # Resample if resample_params includes `sfreq`
            if 'sfreq' in resample_params:
                resample_raw_with_labels(raw_data, **resample_params)

            # Extract events
            events, event_id = extract_events(raw_data)
            # Create epochs with the passed epoch_params
            epochs = mne.Epochs(raw_data, events, event_id, **epoch_params)

            # Exclude bad epochs
            if exclude_bad:
                epochs.drop_bad()

            # Collect index entries for this file
            local_index_data = []

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
                local_index_data.append({
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

            return local_index_data

        # Use batch_process to process data and collect results
        results = self.batch_process(
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
            result_type='value',
            execution_mode='process',
        )

        # Merge index_data from all results in main thread
        index_data = []
        for item in results:
            if item is not None and isinstance(item, list):
                index_data.extend(item)

        # Save index.csv
        pd.DataFrame(index_data).to_csv(index_path, index=False)
        print(f"Index file saved to {index_path}")


    def epoch_by_long_event(self, *args, **kwargs) -> None:
        """
        .. deprecated::
            This method is no longer maintained and will be removed in a future
            release. Use :meth:`epoch_by_long_event_hdf5` instead, which produces
            a single HDF5 file with significantly faster IO and smaller storage.

        Raises
        ------
        NotImplementedError
            Always. Migrate to ``epoch_by_long_event_hdf5``.
        """
        raise NotImplementedError(
            "epoch_by_long_event() (npy output) is no longer maintained. "
            "Use epoch_by_long_event_hdf5() instead: it produces a single HDF5 file "
            "with faster random-access IO and smaller file size. "
            "See the EEGUnity documentation for migration details."
        )


    def epoch_by_event_hdf5(self, output_path: str,
                            exclude_bad: bool = True,
                            file_name_prefix: str = "EpochData",
                            miss_bad_data: bool = False,
                            include_events: Optional[List[str]] = None,
                            format_version: str = 'v2',
                            get_data_row_params: Dict = None,
                            resample_params: Dict = None,
                            epoch_params: Dict = None,
                            pipeline: Optional[Callable] = None) -> None:
        """
        Batch process EEG data to create epochs based on events and save as HDF5.

        **v2 format (default)** — flat array layout optimised for PyTorch random
        access and storage efficiency:

        - ``data`` array: ``(N, n_ch, n_times)`` float32, gzip-1, chunk per epoch.
        - ``epoch_meta/source_group``: source file name for each epoch.
        - ``epoch_meta/event_code``: integer class code for each epoch.
        - ``source_meta/{group}/``: per-file attrs + pickled ``mne.Info``.
        - Root attrs include ``label_map`` (JSON: code → event name).

        **v1 format** — legacy file-per-group layout (deprecated, will be removed
        in a future release).

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Default is ``True``.
        file_name_prefix : str, optional
            Filename prefix for the HDF5 file. Default is ``'EpochData'``.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is ``False``.
        include_events : list of str, optional
            Whitelist of event names to include. If ``None``, all events are
            saved. Use this to exclude noise events (e.g. ``'Start of a trial'``).
        format_version : str, optional
            ``'v2'`` (default, recommended) or ``'v1'`` (deprecated).
        get_data_row_params : dict, optional
            Additional parameters passed to ``get_data_row()``.
        resample_params : dict, optional
            Parameters for resampling. Must include ``sfreq`` for target rate.
        epoch_params : dict, optional
            Additional parameters for ``mne.Epochs``.
        pipeline : callable, optional
            A user-supplied preprocessing function applied to each raw recording
            *after* loading and *before* resampling:
            ``raw = pipeline(raw)``.
            If ``pipeline`` itself performs resampling and ``resample_params``
            is also provided, the final resample (from ``resample_params``)
            will still be applied afterwards — ``resample_params`` always runs
            last. To avoid double-resampling, either omit ``resample_params``
            when the pipeline already resamples, or ensure both target the same
            sampling frequency.

        Raises
        ------
        ValueError
            If recordings have inconsistent channel counts. Use
            ``eeg_batch.auto_domain()`` + ``group_by_domain()`` to split them
            first, then call this function on each sub-dataset separately.
        """
        if get_data_row_params is None:
            get_data_row_params = {}
        if resample_params is None:
            resample_params = {}
        if epoch_params is None:
            epoch_params = {}

        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)

        if format_version == 'v1':
            warnings.warn(_V1_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            self._epoch_by_event_hdf5_v1(
                output_path, exclude_bad, file_name_prefix, miss_bad_data,
                get_data_row_params, resample_params, epoch_params, pipeline)
            return

        # ---- v2 path ----
        _check_channel_consistency(self)

        dataset = h5EpochDatasetV2(Path(output_path), name=file_name_prefix)

        @handle_errors(miss_bad_data)
        @log_processing
        def app_func(row):
            raw_data = self._get_data_row(row, **get_data_row_params)
            if pipeline is not None:
                raw_data = pipeline(raw_data)
            if 'sfreq' in resample_params:
                resample_raw_with_labels(raw_data, **resample_params)

            events, event_id = extract_events(raw_data)
            epochs = mne.Epochs(raw_data, events, event_id, **epoch_params)
            if exclude_bad:
                epochs.drop_bad()

            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            info_bytes = pickle.dumps(raw_data.info)
            source_attrs = _extract_source_attrs(raw_data, row)

            for event in event_id:
                if include_events is not None and event not in include_events:
                    continue
                try:
                    epoch_data = epochs[event].get_data()
                    if epoch_data.ndim != 3 or epoch_data.shape[0] == 0:
                        continue
                    dataset.add_epochs(
                        group_name=file_name,
                        event_name=event,
                        epoch_data=epoch_data,
                        info_bytes=info_bytes,
                        source_attrs=source_attrs,
                        sfreq=raw_data.info['sfreq'],
                        ch_names=raw_data.info['ch_names'],
                    )
                    print(f"  [{file_name}] event='{event}' n={epoch_data.shape[0]}")
                except Exception as e:
                    print(f"  [{file_name}] skipping event '{event}': {e}")

        self.batch_process(
            lambda row: True,
            app_func,
            is_patch=False,
            result_type=None,
            execution_mode=None,
        )

        dataset.save()
        print(f"v2 HDF5 saved to {output_path}/{file_name_prefix}.hdf5")

    def _epoch_by_event_hdf5_v1(self, output_path, exclude_bad, file_name_prefix,
                                 miss_bad_data, get_data_row_params,
                                 resample_params, epoch_params, pipeline=None):
        """Legacy v1 writer (file-per-group layout). Called by epoch_by_event_hdf5."""
        dataset = h5Dataset(Path(output_path), name=file_name_prefix)
        event_info = {}

        @handle_errors(miss_bad_data)
        @log_processing
        def app_func(row):
            raw_data = self._get_data_row(row, **get_data_row_params)
            if pipeline is not None:
                raw_data = pipeline(raw_data)
            if 'sfreq' in resample_params:
                resample_raw_with_labels(raw_data, **resample_params)
            events, event_id = extract_events(raw_data)
            epochs = mne.Epochs(raw_data, events, event_id, **epoch_params)
            if exclude_bad:
                epochs.drop_bad()
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            grp = dataset.addGroup(grpName=file_name)
            info_bytes = pickle.dumps(raw_data.info)
            dataset.addDataset(grp, 'info', np.frombuffer(info_bytes, dtype='uint8'), chunks=None)
            for event in event_id:
                try:
                    event_epochs = epochs[event]
                    epoch_data = event_epochs.get_data()
                    if epoch_data.ndim != 3:
                        raise ValueError("Epoch data is not three-dimensional.")
                    dset = dataset.addDataset(grp, event, epoch_data, chunks=epoch_data.shape)
                    dataset.addAttributes(dset, 'rsFreq', raw_data.info['sfreq'])
                    dataset.addAttributes(dset, 'chOrder', event_epochs.info['ch_names'])
                    event_info[event] = event_info.get(event, 0) + len(event_epochs)
                except Exception as e:
                    print(f"Error processing event '{event}': {e}")

        self.batch_process(
            lambda row: True, app_func,
            is_patch=False, result_type=None, execution_mode=None,
        )
        dataset.save()
        event_info_path = os.path.join(output_path, file_name_prefix + "_event_info.json")
        with open(event_info_path, 'w') as f:
            json.dump(event_info, f, indent=4)
        print(f"v1 HDF5 saved to {output_path}")

    def epoch_by_segmentation_hdf5(self, output_path: str,
                                   exclude_bad: bool = True,
                                   file_name_prefix: str = "EpochData",
                                   miss_bad_data: bool = False,
                                   format_version: str = 'v2',
                                   get_data_row_params: Dict = None,
                                   resample_params: Dict = None,
                                   segment_params: Dict = None,
                                   epoch_params: Dict = None,
                                   pipeline: Optional[Callable] = None) -> None:
        """
        Batch process EEG data to create epochs by sliding-window segmentation
        and save as HDF5.

        See ``epoch_by_event_hdf5`` for documentation of the v2 file format.

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Default is ``True``.
        file_name_prefix : str, optional
            Filename prefix for the HDF5 file. Default is ``'EpochData'``.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is ``False``.
        format_version : str, optional
            ``'v2'`` (default, recommended) or ``'v1'`` (deprecated).
        get_data_row_params : dict, optional
            Additional parameters passed to ``get_data_row()``.
        resample_params : dict, optional
            Parameters for resampling.
        segment_params : dict, optional
            Must include ``'segment_length'`` (seconds) and ``'overlap'`` (0–1).
        epoch_params : dict, optional
            Additional parameters for ``mne.Epochs``.
        pipeline : callable, optional
            Preprocessing function applied after loading and before resampling:
            ``raw = pipeline(raw)``. See ``epoch_by_event_hdf5`` for details on
            the pipeline/resample_params ordering.

        Raises
        ------
        ValueError
            If recordings have inconsistent channel counts. Use
            ``eeg_batch.auto_domain()`` + ``group_by_domain()`` first.
        """
        if get_data_row_params is None:
            get_data_row_params = {}
        if resample_params is None:
            resample_params = {}
        if segment_params is None:
            segment_params = {}
        if epoch_params is None:
            epoch_params = {}

        if 'segment_length' not in segment_params or 'overlap' not in segment_params:
            raise ValueError("segment_params must include 'segment_length' and 'overlap' keys.")

        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)

        if format_version == 'v1':
            warnings.warn(_V1_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            self._epoch_by_segmentation_hdf5_v1(
                output_path, exclude_bad, file_name_prefix, miss_bad_data,
                get_data_row_params, resample_params, segment_params, epoch_params, pipeline)
            return

        # ---- v2 path ----
        _check_channel_consistency(self)

        dataset = h5EpochDatasetV2(Path(output_path), name=file_name_prefix)

        @handle_errors(miss_bad_data)
        @log_processing
        def app_func(row):
            raw_data = self._get_data_row(row, **get_data_row_params)
            if pipeline is not None:
                raw_data = pipeline(raw_data)
            if 'sfreq' in resample_params:
                resample_raw_with_labels(raw_data, **resample_params)

            sfreq = raw_data.info['sfreq']
            segment_length = segment_params['segment_length']
            overlap = segment_params['overlap']
            window_size = int(segment_length * sfreq)
            step_size = int(window_size * (1 - overlap))
            onset_samples = np.arange(0, raw_data.n_times - window_size + 1, step_size)
            n_segments = len(onset_samples)

            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            segment_event_name = 'Segment'

            events = np.zeros((n_segments, 3), dtype=int)
            events[:, 0] = onset_samples
            events[:, 2] = 1
            epoch_id = {segment_event_name: 1}

            epochs = mne.Epochs(raw_data, events, epoch_id,
                                tmin=0, tmax=segment_length, **epoch_params)
            if exclude_bad:
                epochs.drop_bad()

            epoch_data = epochs.get_data()
            if epoch_data.ndim != 3 or epoch_data.shape[0] == 0:
                return

            info_bytes = pickle.dumps(raw_data.info)
            source_attrs = _extract_source_attrs(raw_data, row)
            dataset.add_epochs(
                group_name=file_name,
                event_name=segment_event_name,
                epoch_data=epoch_data,
                info_bytes=info_bytes,
                source_attrs=source_attrs,
                sfreq=sfreq,
                ch_names=raw_data.info['ch_names'],
            )
            print(f"  [{file_name}] {epoch_data.shape[0]} segments saved")

        self.batch_process(
            lambda row: True, app_func,
            is_patch=False, result_type=None, execution_mode=None,
        )
        dataset.save()
        print(f"v2 HDF5 saved to {output_path}/{file_name_prefix}.hdf5")

    def _epoch_by_segmentation_hdf5_v1(self, output_path, exclude_bad,
                                        file_name_prefix, miss_bad_data,
                                        get_data_row_params, resample_params,
                                        segment_params, epoch_params, pipeline=None):
        """Legacy v1 sliding-window writer."""
        dataset = h5Dataset(Path(output_path), name=file_name_prefix)
        event_info = {}

        @handle_errors(miss_bad_data)
        @log_processing
        def app_func(row):
            raw_data = self._get_data_row(row, **get_data_row_params)
            if pipeline is not None:
                raw_data = pipeline(raw_data)
            if 'sfreq' in resample_params:
                resample_raw_with_labels(raw_data, **resample_params)
            sfreq = raw_data.info['sfreq']
            segment_length = segment_params['segment_length']
            overlap = segment_params['overlap']
            window_size = int(segment_length * sfreq)
            step_size = int(window_size * (1 - overlap))
            onset_samples = np.arange(0, raw_data.n_times - window_size + 1, step_size)
            n_segments = len(onset_samples)
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            segment_event_name = f"{file_name}_Segment"
            events = np.zeros((n_segments, 3), dtype=int)
            events[:, 0] = onset_samples
            events[:, 2] = 1
            event_id = {segment_event_name: 1}
            epochs = mne.Epochs(raw_data, events, event_id,
                                tmin=0, tmax=segment_length, **epoch_params)
            if exclude_bad:
                epochs.drop_bad()
            grp = dataset.addGroup(grpName=file_name)
            info_bytes = pickle.dumps(raw_data.info)
            dataset.addDataset(grp, 'info', np.frombuffer(info_bytes, dtype='uint8'), chunks=None)
            try:
                epoch_data = epochs.get_data()
                if epoch_data.ndim != 3:
                    raise ValueError("Epoch data is not three-dimensional.")
                dset = dataset.addDataset(grp, segment_event_name, epoch_data, chunks=epoch_data.shape)
                dataset.addAttributes(dset, 'rsFreq', sfreq)
                dataset.addAttributes(dset, 'chOrder', epochs.info['ch_names'])
                event_info[segment_event_name] = event_info.get(segment_event_name, 0) + len(epochs)
            except Exception as e:
                print(f"Error processing segmented epochs: {e}")

        self.batch_process(
            lambda row: True, app_func,
            is_patch=False, result_type=None, execution_mode=None,
        )
        dataset.save()
        event_info_path = os.path.join(output_path, file_name_prefix + "_event_info.json")
        with open(event_info_path, 'w') as f:
            json.dump(event_info, f, indent=4)
        print(f"v1 HDF5 saved to {output_path}")

    def epoch_by_long_event_hdf5(self, output_path: str,
                                 overlap: float,
                                 file_name_prefix: str = "EpochData",
                                 exclude_bad: bool = True,
                                 miss_bad_data: bool = False,
                                 include_events: Optional[List[str]] = None,
                                 format_version: str = 'v2',
                                 get_data_row_params: Dict = None,
                                 resample_params: Dict = None,
                                 epoch_params: Dict = None,
                                 pipeline: Optional[Callable] = None) -> None:
        """
        Batch process EEG data to create epochs from long-duration events
        (with overlap) and save as HDF5.

        See ``epoch_by_event_hdf5`` for documentation of the v2 file format.

        Parameters
        ----------
        output_path : str
            Directory to save the processed epochs.
        overlap : float
            Overlap between consecutive segments (0.0 ≤ overlap < 1.0).
        file_name_prefix : str, optional
            Filename prefix for the HDF5 file. Default is ``'EpochData'``.
        exclude_bad : bool, optional
            Whether to exclude bad epochs. Default is ``True``.
        miss_bad_data : bool, optional
            Whether to skip files with processing errors. Default is ``False``.
        include_events : list of str, optional
            Whitelist of event names to include. ``None`` keeps all events.
        format_version : str, optional
            ``'v2'`` (default, recommended) or ``'v1'`` (deprecated).
        get_data_row_params : dict, optional
            Additional parameters passed to ``get_data_row()``.
        resample_params : dict, optional
            Parameters for resampling.
        epoch_params : dict, optional
            Additional parameters for ``mne.Epochs``.
        pipeline : callable, optional
            Preprocessing function applied after loading and before resampling:
            ``raw = pipeline(raw)``. See ``epoch_by_event_hdf5`` for details on
            the pipeline/resample_params ordering.

        Raises
        ------
        ValueError
            If recordings have inconsistent channel counts. Use
            ``eeg_batch.auto_domain()`` + ``group_by_domain()`` first.
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

        if format_version == 'v1':
            warnings.warn(_V1_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            self._epoch_by_long_event_hdf5_v1(
                output_path, overlap, exclude_bad, file_name_prefix, miss_bad_data,
                get_data_row_params, resample_params, epoch_params, pipeline)
            return

        # ---- v2 path ----
        _check_channel_consistency(self)

        dataset = h5EpochDatasetV2(Path(output_path), name=file_name_prefix)

        @handle_errors(miss_bad_data)
        def process_file(row):
            raw_data = self._get_data_row(row, **get_data_row_params)
            if pipeline is not None:
                raw_data = pipeline(raw_data)
            if 'sfreq' in resample_params:
                resample_raw_with_labels(raw_data, **resample_params)

            annotations = raw_data.annotations
            if not annotations:
                raise ValueError(f"No annotations in {row['File Path']}.")

            t_min = epoch_params.get('tmin', 0)
            t_max = epoch_params.get('tmax', 1)
            epoch_length = t_max - t_min

            temp_annotations = []
            for onset, duration, description in zip(
                    annotations.onset, annotations.duration, annotations.description):
                if duration > epoch_length:
                    step = epoch_length * (1 - overlap)
                    num_segments = int((duration - epoch_length) // step) + 1
                    for i in range(num_segments):
                        temp_annotations.append(
                            (onset + i * step, epoch_length, description))

            if not temp_annotations:
                raise ValueError("No long events long enough for segmentation.")

            raw_data.set_annotations(mne.Annotations(
                onset=[a[0] for a in temp_annotations],
                duration=[a[1] for a in temp_annotations],
                description=[a[2] for a in temp_annotations],
            ))
            events, event_id_map = mne.events_from_annotations(raw_data)
            epochs = mne.Epochs(raw_data, events, event_id_map, **epoch_params)
            if exclude_bad:
                epochs.drop_bad()

            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            info_bytes = pickle.dumps(raw_data.info)
            source_attrs = _extract_source_attrs(raw_data, row)

            for description in event_id_map:
                if include_events is not None and description not in include_events:
                    continue
                try:
                    epoch_data = epochs[description].get_data()
                    if epoch_data.ndim != 3 or epoch_data.shape[0] == 0:
                        continue
                    dataset.add_epochs(
                        group_name=file_name,
                        event_name=description,
                        epoch_data=epoch_data,
                        info_bytes=info_bytes,
                        source_attrs=source_attrs,
                        sfreq=raw_data.info['sfreq'],
                        ch_names=raw_data.info['ch_names'],
                    )
                    print(f"  [{file_name}] event='{description}' n={epoch_data.shape[0]}")
                except Exception as e:
                    print(f"  [{file_name}] skipping event '{description}': {e}")

        self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: process_file(row),
            is_patch=False, result_type=None, execution_mode=None,
        )
        dataset.save()
        print(f"v2 HDF5 saved to {output_path}/{file_name_prefix}.hdf5")

    def _epoch_by_long_event_hdf5_v1(self, output_path, overlap, exclude_bad,
                                      file_name_prefix, miss_bad_data,
                                      get_data_row_params, resample_params, epoch_params,
                                      pipeline=None):
        """Legacy v1 long-event writer."""
        dataset = h5Dataset(Path(output_path), name=file_name_prefix)
        event_info = {}

        @handle_errors(miss_bad_data)
        def process_file(row):
            raw_data = self._get_data_row(row, **get_data_row_params)
            if pipeline is not None:
                raw_data = pipeline(raw_data)
            if 'sfreq' in resample_params:
                resample_raw_with_labels(raw_data, **resample_params)
            annotations = raw_data.annotations
            if not annotations:
                raise ValueError(f"No annotations in {row['File Path']}.")
            t_min = epoch_params.get('tmin', 0)
            t_max = epoch_params.get('tmax', 1)
            epoch_length = t_max - t_min
            temp_annotations = []
            for onset, duration, description in zip(
                    annotations.onset, annotations.duration, annotations.description):
                if duration > epoch_length:
                    step = epoch_length * (1 - overlap)
                    num_segments = int((duration - epoch_length) // step) + 1
                    for i in range(num_segments):
                        temp_annotations.append((onset + i * step, epoch_length, description))
            if not temp_annotations:
                raise ValueError("No long events long enough for segmentation.")
            raw_data.set_annotations(mne.Annotations(
                onset=[a[0] for a in temp_annotations],
                duration=[a[1] for a in temp_annotations],
                description=[a[2] for a in temp_annotations],
            ))
            events, event_id_map = mne.events_from_annotations(raw_data)
            epochs = mne.Epochs(raw_data, events, event_id_map, **epoch_params)
            if exclude_bad:
                epochs.drop_bad()
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            grp = dataset.addGroup(grpName=file_name)
            info_bytes = pickle.dumps(raw_data.info)
            dataset.addDataset(grp, 'info', np.frombuffer(info_bytes, dtype='uint8'), chunks=None)
            for description, event_id in event_id_map.items():
                try:
                    event_epochs = epochs[description]
                    epoch_data = event_epochs.get_data()
                    if epoch_data.ndim != 3:
                        raise ValueError("Not 3D.")
                    dset = dataset.addDataset(grp, description, epoch_data, chunks=epoch_data.shape)
                    dataset.addAttributes(dset, 'rsFreq', raw_data.info['sfreq'])
                    dataset.addAttributes(dset, 'chOrder', event_epochs.info['ch_names'])
                    event_info[description] = event_info.get(description, 0) + len(event_epochs)
                except Exception as e:
                    print(f"Error processing event '{description}': {e}")

        self.batch_process(
            lambda row: row['Completeness Check'] != 'Unavailable',
            lambda row: process_file(row),
            is_patch=False, result_type=None, execution_mode=None,
        )
        dataset.save()
        event_info_path = os.path.join(output_path, file_name_prefix + "_event_info.json")
        with open(event_info_path, 'w') as f:
            json.dump(event_info, f, indent=4)
        print(f"v1 HDF5 saved to {output_path}.")


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

        Method selection rule:
        ``(long_event=False, use_hdf5=False) -> epoch_by_event``;
        ``(long_event=False, use_hdf5=True) -> epoch_by_event_hdf5``;
        ``(long_event=True, use_hdf5=False) -> epoch_by_long_event``;
        ``(long_event=True, use_hdf5=True) -> epoch_by_long_event_hdf5``.

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
            Additional parameters for data retrieval via ``get_data_row()``.
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


# ---------------------------------------------------------------------------
# Module-level helpers (not methods — avoid cluttering the mixin namespace)
# ---------------------------------------------------------------------------

def _check_channel_consistency(batch_obj) -> None:
    """Raise ValueError if the locator contains files with differing channel counts."""
    locator = batch_obj.get_shared_attr()['locator']
    available = locator[locator['Completeness Check'] != 'Unavailable']
    unique_counts = available['Number of Channels'].astype(float).astype(int).unique()
    if len(unique_counts) > 1:
        raise ValueError(
            f"Inconsistent channel counts detected: {sorted(unique_counts.tolist())}. "
            "All recordings must have the same number of channels before exporting to "
            "a flat HDF5 (v2) file. "
            "Use eeg_batch.auto_domain() followed by group_by_domain() to split "
            "recordings with different channel configurations into separate "
            "UnifiedDataset instances, then call this function on each one."
        )


def _extract_source_attrs(raw, row) -> dict:
    """Extract scalar metadata from a raw object and locator row."""
    attrs = {'file_path': str(row.get('File Path', 'unknown'))}
    try:
        desc = json.loads(raw.info.get('description', '{}'))
        eu = desc.get('eegunity_description', {})
        attrs['age'] = eu.get('age', 'unknown')
        attrs['gender'] = eu.get('sex', 'unknown')
        attrs['amplifier'] = eu.get('amplifier', 'unknown')
        attrs['cap'] = eu.get('cap', 'unknown')
        attrs['handedness'] = eu.get('handedness', 'unknown')
    except Exception:
        for key in ('age', 'gender', 'amplifier', 'cap', 'handedness'):
            attrs[key] = 'unknown'
    return attrs
