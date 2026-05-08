import json
import datetime
import h5py
import numpy as np
from pathlib import Path


class h5Dataset:
    """
   Handle HDF5 file operations in a format compatible with h5py.

    This class is adapted from:
    https://github.com/935963004/LaBraM/blob/main/dataset_maker/shock/utils/h5.py#L8.

    Atomic write
    ------------
    Data is written to ``<name>.hdf5.tmp`` and only renamed to
    ``<name>.hdf5`` when :meth:`save` is called successfully.
    If the final ``.hdf5`` file already exists, ``FileExistsError``
    is raised at construction time so the user can clean up explicitly.
    """

    def __init__(self, path: Path, name: str) -> None:
        """
        Initialize the HDF5 dataset handler.

        Parameters
        ----------
        path : Path
            The path to the directory containing the HDF5 file.
        name : str
            The name of the HDF5 file (without extension).

        Raises
        ------
        FileExistsError
            If ``<path>/<name>.hdf5`` already exists.
        """
        self.__name = name
        self.__final_path = Path(path) / f'{name}.hdf5'
        self.__tmp_path = Path(path) / f'{name}.hdf5.tmp'

        if self.__final_path.exists():
            raise FileExistsError(
                f"HDF5 file already exists: {self.__final_path}. "
                "Delete it manually before re-exporting."
            )
        if self.__tmp_path.exists():
            self.__tmp_path.unlink()

        self.__f = h5py.File(self.__tmp_path, 'w')

    def addGroup(self, grpName: str):
        """
        Add a new group to the HDF5 file.

        Parameters
        ----------
        grpName : str
            The name of the group to create.

        Returns
        -------
        h5py.Group
            The created group object.
        """
        return self.__f.create_group(grpName)

    def addDataset(self, grp: h5py.Group, dsName: str, arr: np.array, chunks: tuple = None, **kwargs):
        """
        Add a dataset to a specified group.

        Parameters
        ----------
        grp : h5py.Group
            The group to which the dataset will be added.
        dsName : str
            The name of the dataset.
        arr : np.array
            The data to store in the dataset.
        chunks : tuple, optional
            The chunk shape to use when storing the dataset.
        **kwargs
            Additional keyword arguments passed to `create_dataset`.

        Returns
        -------
        h5py.Dataset
            The created dataset object.
        """
        return grp.create_dataset(dsName, data=arr, chunks=chunks, **kwargs)

    def addAttributes(self, src: 'h5py.Dataset|h5py.Group', attrName: str, attrValue):
        """
        Add an attribute to a dataset or group.

        Parameters
        ----------
        src : h5py.Dataset or h5py.Group
            The target object to which the attribute will be added.
        attrName : str
            The name of the attribute.
        attrValue : any
            The value of the attribute.
        """
        src.attrs[f'{attrName}'] = attrValue

    def save(self):
        """Close the tmp file and atomically rename it to the final path."""
        self.__f.close()
        self.__tmp_path.rename(self.__final_path)

    @property
    def name(self):
        """
        Get the name of the HDF5 dataset.

        Returns
        -------
        str
            The name of the HDF5 file.
        """
        return self.__name


class h5EpochDatasetV2:
    """
    EEGUnity v2 epoch HDF5 format writer (v2.1 schema).

    Flat-array layout optimised for PyTorch random access:

    Structure
    ---------
    / (root)
      attrs: version ("2.1"), sfreq, ch_names (JSON), n_channels, n_times,
             label_map (JSON: code->event_name), n_epochs_total,
             info_fields (JSON list[str]; v2.1),
             created_by, created_at
    ├── data          (N, n_ch, n_times)  float32
    │                 chunk=(1, n_ch, n_times), gzip level-1
    ├── epoch_meta/
    │   ├── source_group  (N,)  variable-length UTF-8 string
    │   └── event_code    (N,)  int16
    ├── misc_meta/                                              (v2.1, optional)
    │   ├── {misc_channel_name}  (N,) float32
    │   └── attrs.names: JSON list[str]
    └── source_meta/
        └── {group_name}/
              attrs: file_path, n_epochs_in_source, sfreq,
                     ch_names (JSON), age, gender, amplifier, cap, handedness,
                     <any additional participants.tsv columns>
              └── info  (uint8)  pickled mne.Info bytes

    Backward compat
    ---------------
    Readers of v2.0 can still open v2.1 files if they ignore /misc_meta/ and the
    extra source attrs. The dataset_api in eeg_kernel_agent accepts both "2.0"
    and "2.1" as valid version strings.

    Usage
    -----
    writer = h5EpochDatasetV2(output_dir, "MyDataset")
    writer.add_epochs(group_name, event_name, epoch_array_float32,
                      info_bytes, source_attrs, sfreq, ch_names,
                      misc_values={"accuracy": np.array([1, 0, 1])})   # v2.1
    writer.save()

    Reading in PyTorch
    ------------------
    Use eeg_kernel_agent.dataset_api.{EventDataset, MISCDataset, InfoDataset}.
    """

    def __init__(self, path: Path, name: str) -> None:
        self._final_path = Path(path) / f'{name}.hdf5'
        self._tmp_path = Path(path) / f'{name}.hdf5.tmp'

        if self._final_path.exists():
            raise FileExistsError(
                f"HDF5 file already exists: {self._final_path}. "
                "Delete it manually before re-exporting."
            )
        if self._tmp_path.exists():
            self._tmp_path.unlink()

        self._f = None
        self._label_map: dict = {}   # event_name -> int code
        self._next_code: int = 0
        self._n_ch: int = None
        self._n_times: int = None
        self._source_epoch_counts: dict = {}  # group_name -> cumulative count
        self._info_fields: set = set()        # v2.1: non-canonical source attr keys
        self._misc_names: list = []           # v2.1: preserves insertion order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self, n_ch: int, n_times: int,
                             sfreq: float, ch_names) -> None:
        if self._f is not None:
            return
        self._n_ch = n_ch
        self._n_times = n_times
        self._f = h5py.File(self._tmp_path, 'w')

        self._f.attrs['version'] = '2.1'
        self._f.attrs['sfreq'] = float(sfreq)
        self._f.attrs['ch_names'] = json.dumps(list(ch_names))
        self._f.attrs['n_channels'] = int(n_ch)
        self._f.attrs['n_times'] = int(n_times)
        self._f.attrs['created_by'] = 'EEGUnity'
        self._f.attrs['created_at'] = datetime.datetime.now().isoformat()

        self._f.create_dataset(
            'data',
            shape=(0, n_ch, n_times),
            maxshape=(None, n_ch, n_times),
            dtype='float32',
            chunks=(1, n_ch, n_times),
            compression='gzip',
            compression_opts=1,
        )

        em = self._f.create_group('epoch_meta')
        em.create_dataset(
            'source_group',
            shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype(encoding='utf-8'),
        )
        em.create_dataset(
            'event_code',
            shape=(0,), maxshape=(None,),
            dtype='int16',
        )

        self._f.create_group('source_meta')

    def _get_or_create_code(self, event_name: str) -> int:
        if event_name not in self._label_map:
            self._label_map[event_name] = self._next_code
            self._next_code += 1
        return self._label_map[event_name]

    def _append_1d(self, ds_name: str, values) -> None:
        ds = self._f[ds_name]
        old = ds.shape[0]
        n = len(values)
        ds.resize(old + n, axis=0)
        ds[old:old + n] = values

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _CANONICAL_SOURCE_ATTRS = (
        'file_path', 'age', 'gender', 'amplifier', 'cap', 'handedness',
    )

    def add_epochs(
        self,
        group_name: str,
        event_name: str,
        epoch_data: np.ndarray,
        info_bytes: bytes,
        source_attrs: dict,
        sfreq: float,
        ch_names,
        misc_values: dict = None,
    ) -> None:
        """
        Append epochs for one (source_file, event) pair.

        Parameters
        ----------
        group_name : str
            Unique identifier for the source file (e.g. basename without extension).
        event_name : str
            Human-readable event / class label.
        epoch_data : np.ndarray, shape (n_epochs, n_ch, n_times)
            Epoch array; will be cast to float32.
        info_bytes : bytes
            ``pickle.dumps(raw.info)`` from the source file.
        source_attrs : dict
            Scalar metadata stored as HDF5 attrs on the source_meta group.
            Canonical keys (optional): file_path, age, gender, amplifier,
            cap, handedness. Any additional keys (e.g. participants.tsv
            columns like ``p_factor``) are stored verbatim and their names
            recorded in the root ``info_fields`` attr.
        sfreq : float
            Sampling frequency (used only during lazy initialisation).
        ch_names : list[str]
            Channel names (used only during lazy initialisation).
        misc_values : dict, optional
            Mapping ``{misc_channel_name: array-like of length n}`` for v2.1
            per-epoch misc values (e.g. reaction_time, accuracy). Stored
            under ``/misc_meta/<name>`` as float32.
        """
        n, n_ch, n_times = epoch_data.shape
        if n == 0:
            return

        self._ensure_initialized(n_ch, n_times, sfreq, ch_names)

        if n_ch != self._n_ch or n_times != self._n_times:
            raise ValueError(
                f"Epoch shape mismatch for group '{group_name}', event '{event_name}': "
                f"expected ({self._n_ch}, {self._n_times}), got ({n_ch}, {n_times}). "
                "Ensure all recordings have the same channel count and epoch length."
            )

        # ---- Register source_meta (once per source file) ----
        sm = self._f['source_meta']
        if group_name not in sm:
            grp = sm.create_group(group_name)
            grp.create_dataset(
                'info',
                data=np.frombuffer(info_bytes, dtype='uint8'),
                chunks=None,
            )
            grp.attrs['sfreq'] = float(sfreq)
            grp.attrs['ch_names'] = json.dumps(list(ch_names))
            for key in self._CANONICAL_SOURCE_ATTRS:
                val = source_attrs.get(key, 'unknown')
                grp.attrs[key] = str(val) if val is not None else 'unknown'
            # v2.1: persist arbitrary extra attrs (participants.tsv columns etc.)
            for key, val in source_attrs.items():
                if key in self._CANONICAL_SOURCE_ATTRS:
                    continue
                if val is None:
                    continue
                grp.attrs[key] = str(val)
                self._info_fields.add(key)

        # ---- Append epoch data ----
        old_size = self._f['data'].shape[0]
        self._f['data'].resize(old_size + n, axis=0)
        self._f['data'][old_size:old_size + n] = epoch_data.astype('float32')

        # ---- Append epoch_meta ----
        code = self._get_or_create_code(event_name)
        self._append_1d('epoch_meta/source_group', np.array([group_name] * n, dtype=object))
        self._append_1d('epoch_meta/event_code', np.full(n, code, dtype='int16'))

        # ---- Append misc_meta (v2.1) ----
        # Always call if *any* misc channel has been registered, so newly-added
        # channels get NaN padding for all prior epochs.
        if misc_values or self._misc_names:
            self._append_misc(misc_values or {}, n, group_name, event_name)

        # ---- Update per-source epoch count ----
        self._source_epoch_counts[group_name] = (
            self._source_epoch_counts.get(group_name, 0) + n
        )

    def _append_misc(self, misc_values: dict, n: int,
                     group_name: str, event_name: str) -> None:
        if 'misc_meta' not in self._f:
            self._f.create_group('misc_meta')
        mm = self._f['misc_meta']
        current_total = self._f['data'].shape[0]  # already resized
        old_total = current_total - n

        for name, values in misc_values.items():
            arr = np.asarray(values, dtype='float32')
            if arr.shape != (n,):
                raise ValueError(
                    f"misc_values['{name}'] length {arr.shape} does not match "
                    f"epoch count {n} for group '{group_name}', event '{event_name}'."
                )
            if name not in mm:
                mm.create_dataset(
                    name,
                    shape=(old_total,), maxshape=(None,),
                    dtype='float32',
                    fillvalue=float('nan'),
                )
                self._misc_names.append(name)
            ds = mm[name]
            # Pad any prior epochs that lacked this misc channel with NaN.
            if ds.shape[0] < old_total:
                ds.resize(old_total, axis=0)
            ds.resize(old_total + n, axis=0)
            ds[old_total:old_total + n] = arr

        # Other misc channels not supplied this call: extend with NaN so the
        # final array length stays aligned with /data.
        for name in self._misc_names:
            if name in misc_values:
                continue
            ds = mm[name]
            if ds.shape[0] < old_total + n:
                ds.resize(old_total + n, axis=0)

    def save(self) -> None:
        """Finalise the tmp file, then atomically rename it to the final path."""
        if self._f is None:
            return

        # Reverse map: int code -> event_name string
        reverse_map = {int(v): k for k, v in self._label_map.items()}
        self._f.attrs['label_map'] = json.dumps(reverse_map)
        self._f.attrs['n_epochs_total'] = int(self._f['data'].shape[0])
        # v2.1: non-canonical source attr keys (participants.tsv columns etc.)
        self._f.attrs['info_fields'] = json.dumps(sorted(self._info_fields))

        # Persist per-source epoch counts into source_meta attrs
        sm = self._f['source_meta']
        for grp_name, count in self._source_epoch_counts.items():
            if grp_name in sm:
                sm[grp_name].attrs['n_epochs_in_source'] = int(count)

        # v2.1: persist misc channel order on /misc_meta
        if 'misc_meta' in self._f:
            self._f['misc_meta'].attrs['names'] = json.dumps(self._misc_names)

        self._f.close()
        self._f = None
        self._tmp_path.rename(self._final_path)

    @property
    def name(self) -> str:
        return self._final_path.stem
