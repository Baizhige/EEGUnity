"""EEGUnity HDF5 v2.3 materialisation writer and NumPy reader."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


UTF8 = h5py.string_dtype(encoding="utf-8")
VLEN_UINT8 = h5py.vlen_dtype(np.dtype("uint8"))


def _field_name(value) -> str:
    name = str(value).strip()
    if not name or "/" in name:
        raise ValueError(f"HDF5 metadata field must be non-empty and contain no '/': {value!r}")
    return name


class h5EpochDatasetV23:
    """Write an immutable, training-first EEGUnity HDF5 v2.3 shard.

    Signal materialisation is independent from downstream task views. Epochs
    refer to sources by integer ID; paths and Info values are stored once in
    columnar source tables. Long epochs are time-tiled for direct window reads.
    """

    def __init__(
        self,
        path: Path,
        name: str,
        storage_window_samples: int = 800,
        root_attrs: Optional[dict] = None,
        validate_data: bool = True,
    ) -> None:
        self._final_path = Path(path) / f"{name}.hdf5"
        self._tmp_path = Path(path) / f"{name}.hdf5.tmp"
        if self._final_path.exists():
            raise FileExistsError(
                f"HDF5 file already exists: {self._final_path}. "
                "Delete it manually before re-exporting."
            )
        if self._tmp_path.exists():
            self._tmp_path.unlink()
        if int(storage_window_samples) <= 0:
            raise ValueError("storage_window_samples must be positive.")

        self._name = str(name)
        self._storage_window_samples = int(storage_window_samples)
        self._root_attrs = dict(root_attrs or {})
        self._validate_data = bool(validate_data)
        self._f = None
        self._n_ch = None
        self._n_times = None
        self._ch_names = None
        self._ch_types = None
        self._label_map: Dict[str, int] = {}
        self._source_lookup: Dict[str, int] = {}
        self._source_epoch_counts: List[int] = []
        self._source_mask_ids: List[int] = []
        self._mask_lookup: Dict[bytes, int] = {}
        self._misc_names: List[str] = []
        self._epoch_meta_names: List[str] = []

    @staticmethod
    def _normalise_path(value) -> str:
        text = os.path.normpath(str(value or "")).replace(os.sep, "/")
        return text if text not in {"", "."} else "unknown"

    @classmethod
    def _source_uid(cls, file_path: str, source_name: str) -> str:
        canonical = cls._normalise_path(file_path)
        if canonical == "unknown":
            canonical = f"unresolved:{source_name}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _append(dataset: h5py.Dataset, values) -> None:
        old = dataset.shape[0]
        dataset.resize(old + len(values), axis=0)
        dataset[old:] = values

    @staticmethod
    def _vector(values, size: int, dtype, fill_value):
        if values is None:
            return np.full(size, fill_value, dtype=dtype)
        array = np.asarray(values, dtype=dtype)
        if array.ndim == 0:
            array = np.full(size, array.item(), dtype=dtype)
        if array.shape != (size,):
            raise ValueError(
                f"metadata length mismatch: expected ({size},), got {array.shape}."
            )
        return array

    def _ensure_initialized(
        self,
        n_ch: int,
        n_times: int,
        sfreq: float,
        ch_names,
        ch_types=None,
    ) -> None:
        if self._f is not None:
            return
        self._n_ch = int(n_ch)
        self._n_times = int(n_times)
        self._ch_names = list(ch_names)
        self._ch_types = list(ch_types or ["unknown"] * n_ch)
        if len(self._ch_names) != n_ch:
            raise ValueError("ch_names length must match n_ch.")
        if len(self._ch_types) != n_ch:
            raise ValueError("ch_types length must match n_ch.")

        self._tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self._tmp_path, "w")
        time_chunk = (
            n_times
            if n_times <= 2 * self._storage_window_samples
            else min(n_times, self._storage_window_samples)
        )
        self._f.attrs.update(
            {
                "version": "2.3",
                "layout": "materialization-shard",
                "state": "building",
                "materialization_id": self._name,
                "shard_id": "part-00000",
                "sfreq": float(sfreq),
                "n_channels": int(n_ch),
                "n_times": int(n_times),
                "data_dtype": "float32",
                "storage_profile": f"epoch-time{time_chunk}-gzip1-shuffle",
                "ch_names": json.dumps(self._ch_names),
                "ch_types": json.dumps(self._ch_types),
                "created_by": "EEGUnity",
                "created_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
        )
        for key, value in self._root_attrs.items():
            if key in {"version", "layout", "state"}:
                raise ValueError(f"root_attrs cannot override reserved key {key!r}.")
            self._f.attrs[str(key)] = value

        self._f.create_dataset(
            "data",
            shape=(0, n_ch, n_times),
            maxshape=(None, n_ch, n_times),
            dtype="float32",
            chunks=(1, n_ch, time_chunk),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        channels = self._f.create_group("channels")
        channels.create_dataset(
            "name", data=np.asarray(self._ch_names, dtype=object), dtype=UTF8
        )
        channels.create_dataset(
            "type", data=np.asarray(self._ch_types, dtype=object), dtype=UTF8
        )

        epochs = self._f.create_group("epochs")
        one_d = {
            "maxshape": (None,),
            "chunks": (16384,),
            "compression": "gzip",
            "compression_opts": 1,
            "shuffle": True,
        }
        epochs.create_dataset("source_id", shape=(0,), dtype="uint32", **one_d)
        epochs.create_dataset(
            "source_epoch_index", shape=(0,), dtype="uint32", **one_d
        )
        epochs.create_dataset(
            "source_start_sample", shape=(0,), dtype="int64", **one_d
        )
        epochs.create_dataset("event_code", shape=(0,), dtype="int32", **one_d)
        epochs.create_group("meta")
        epochs.create_group("misc")
        self._f.create_group("events")

        sources = self._f.create_group("sources")
        sources.create_dataset("uid", shape=(0,), maxshape=(None,), dtype=UTF8)
        sources.create_dataset("name", shape=(0,), maxshape=(None,), dtype=UTF8)
        sources.create_dataset(
            "file_path", shape=(0,), maxshape=(None,), dtype=UTF8
        )
        sources.create_dataset(
            "sfreq", shape=(0,), maxshape=(None,), dtype="float64"
        )
        sources.create_dataset(
            "n_epochs", shape=(0,), maxshape=(None,), dtype="uint64"
        )
        sources.create_dataset(
            "channel_mask_id", shape=(0,), maxshape=(None,), dtype="uint32"
        )
        sources.create_dataset(
            "info_pickle", shape=(0,), maxshape=(None,), dtype=VLEN_UINT8
        )
        sources.create_group("info")

        masks = self._f.create_group("channel_masks")
        masks.create_dataset(
            "value",
            shape=(0, n_ch),
            maxshape=(None, n_ch),
            dtype="uint8",
            chunks=(1, n_ch),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

    def _register_mask(self, channel_mask) -> int:
        if channel_mask is None:
            mask = np.ones(self._n_ch, dtype="uint8")
        else:
            mask = np.asarray(channel_mask, dtype="uint8")
            if mask.shape != (self._n_ch,):
                raise ValueError(
                    f"channel_mask must have shape ({self._n_ch},), "
                    f"got {mask.shape}."
                )
        key = mask.tobytes()
        mask_id = self._mask_lookup.get(key)
        if mask_id is None:
            mask_id = len(self._mask_lookup)
            self._mask_lookup[key] = mask_id
            dataset = self._f["channel_masks/value"]
            dataset.resize(mask_id + 1, axis=0)
            dataset[mask_id] = mask
        return mask_id

    def _register_source(
        self,
        source_name: str,
        info_bytes: bytes,
        source_attrs: dict,
        sfreq: float,
        channel_mask=None,
    ) -> int:
        source_attrs = dict(source_attrs or {})
        file_path = self._normalise_path(source_attrs.get("file_path", "unknown"))
        uid = self._source_uid(file_path, source_name)
        if uid in self._source_lookup:
            source_id = self._source_lookup[uid]
            mask_id = self._register_mask(channel_mask)
            if self._source_mask_ids[source_id] != mask_id:
                raise ValueError(
                    f"source {file_path!r} changed channel_mask between blocks."
                )
            return source_id

        source_id = len(self._source_lookup)
        self._source_lookup[uid] = source_id
        mask_id = self._register_mask(channel_mask)
        self._source_mask_ids.append(mask_id)
        self._source_epoch_counts.append(0)
        sources = self._f["sources"]
        scalar_values = {
            "uid": uid,
            "name": str(source_name),
            "file_path": file_path,
            "sfreq": float(sfreq),
            "n_epochs": np.uint64(0),
            "channel_mask_id": np.uint32(mask_id),
        }
        for name, value in scalar_values.items():
            dataset = sources[name]
            dataset.resize(source_id + 1, axis=0)
            dataset[source_id] = value
        pickle_dataset = sources["info_pickle"]
        pickle_dataset.resize(source_id + 1, axis=0)
        pickle_dataset[source_id] = np.frombuffer(info_bytes or b"", dtype="uint8")

        info = sources["info"]
        fields = {
            _field_name(key): "" if value is None else str(value)
            for key, value in source_attrs.items()
            if key != "file_path"
        }
        for name in fields:
            if name not in info:
                dataset = info.create_dataset(
                    name, shape=(source_id,), maxshape=(None,), dtype=UTF8
                )
                if source_id:
                    dataset[:] = np.full(source_id, "", dtype=object)
        for name, dataset in info.items():
            dataset.resize(source_id + 1, axis=0)
            dataset[source_id] = fields.get(name, "")
        return source_id

    def _event_code(self, event_name: str) -> int:
        name = str(event_name)
        if name not in self._label_map:
            self._label_map[name] = len(self._label_map)
        return self._label_map[name]

    def _append_epoch_meta(self, values: dict, size: int, old_size: int) -> None:
        group = self._f["epochs/meta"]
        values = {_field_name(key): value for key, value in dict(values or {}).items()}
        for name in values:
            if name not in group:
                dataset = group.create_dataset(
                    name, shape=(old_size,), maxshape=(None,), dtype=UTF8
                )
                if old_size:
                    dataset[:] = np.full(old_size, "", dtype=object)
                self._epoch_meta_names.append(name)
        for name in self._epoch_meta_names:
            raw = values.get(name)
            if raw is None:
                array = np.full(size, "", dtype=object)
            else:
                array = np.asarray(raw, dtype=object)
                if array.ndim == 0:
                    array = np.full(size, array.item(), dtype=object)
                if array.shape != (size,):
                    raise ValueError(
                        f"epoch_meta[{name!r}] must have length {size}."
                    )
                array = np.asarray(
                    ["" if item is None else str(item) for item in array],
                    dtype=object,
                )
            self._append(group[name], array)

    def _append_misc(self, values: dict, size: int, old_size: int) -> None:
        group = self._f["epochs/misc"]
        values = {_field_name(key): value for key, value in dict(values or {}).items()}
        for name in values:
            if name not in group:
                dataset = group.create_dataset(
                    name,
                    shape=(old_size,),
                    maxshape=(None,),
                    dtype="float32",
                    chunks=(16384,),
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                    fillvalue=np.nan,
                )
                if old_size:
                    dataset[:] = np.full(old_size, np.nan, dtype="float32")
                self._misc_names.append(name)
        for name in self._misc_names:
            array = self._vector(values.get(name), size, "float32", np.nan)
            self._append(group[name], array)

    def add_epochs(
        self,
        group_name: str,
        event_name: str,
        epoch_data: np.ndarray,
        info_bytes: bytes,
        source_attrs: dict,
        sfreq: float,
        ch_names,
        misc_values: Optional[dict] = None,
        *,
        ch_types=None,
        epoch_meta: Optional[dict] = None,
        misc_meta: Optional[dict] = None,
        channel_mask=None,
        source_start_samples=None,
    ) -> None:
        """Append one source/event block without repeating source strings."""
        data = np.asarray(epoch_data)
        if data.ndim != 3:
            raise ValueError(f"epoch_data must be rank 3, got {data.shape}.")
        size, n_ch, n_times = data.shape
        if size == 0:
            return
        self._ensure_initialized(n_ch, n_times, sfreq, ch_names, ch_types)
        if (n_ch, n_times) != (self._n_ch, self._n_times):
            raise ValueError(
                f"Epoch shape mismatch: expected ({self._n_ch}, {self._n_times}), "
                f"got ({n_ch}, {n_times})."
            )
        if list(ch_names) != self._ch_names:
            raise ValueError("Channel names/order changed between epoch blocks.")
        if ch_types is not None and list(ch_types) != self._ch_types:
            raise ValueError("Channel types changed between epoch blocks.")
        data = data.astype("float32", copy=False)
        if self._validate_data:
            if not np.isfinite(data).all():
                source = source_attrs.get("file_path", group_name)
                raise ValueError(f"Non-finite EEG samples in source {source!r}.")
            if not np.any(data != 0):
                source = source_attrs.get("file_path", group_name)
                raise ValueError(f"All-zero EEG block in source {source!r}.")

        source_id = self._register_source(
            group_name, info_bytes, source_attrs, sfreq, channel_mask
        )
        source_first = self._source_epoch_counts[source_id]
        starts = self._vector(source_start_samples, size, "int64", -1)
        code = self._event_code(event_name)
        old_size = self._f["data"].shape[0]
        self._f["data"].resize(old_size + size, axis=0)
        self._f["data"][old_size:] = data
        self._append(
            self._f["epochs/source_id"],
            np.full(size, source_id, dtype="uint32"),
        )
        self._append(
            self._f["epochs/source_epoch_index"],
            np.arange(source_first, source_first + size, dtype="uint32"),
        )
        self._append(self._f["epochs/source_start_sample"], starts)
        self._append(
            self._f["epochs/event_code"], np.full(size, code, dtype="int32")
        )
        self._append_epoch_meta(epoch_meta or {}, size, old_size)
        merged_misc = dict(misc_values or {})
        merged_misc.update(dict(misc_meta or {}))
        self._append_misc(merged_misc, size, old_size)
        self._source_epoch_counts[source_id] += size
        self._f["sources/n_epochs"][source_id] = self._source_epoch_counts[source_id]

    def save(self) -> None:
        """Finalize, fsync, and atomically publish the v2.3 file."""
        if self._f is None or self._f["data"].shape[0] == 0:
            raise RuntimeError("No epochs were written to the HDF5 export.")
        event_names = [""] * len(self._label_map)
        for name, code in self._label_map.items():
            event_names[code] = name
        self._f["events"].create_dataset(
            "name", data=np.asarray(event_names, dtype=object), dtype=UTF8
        )
        reverse_map = {int(code): name for name, code in self._label_map.items()}
        self._f.attrs["label_map"] = json.dumps(reverse_map)
        self._f.attrs["n_epochs_total"] = int(self._f["data"].shape[0])
        self._f.attrs["n_sources"] = len(self._source_lookup)
        self._f.attrs["info_fields"] = json.dumps(
            sorted(self._f["sources/info"].keys())
        )
        self._f.attrs["misc_fields"] = json.dumps(
            sorted(self._f["epochs/misc"].keys())
        )
        self._f.attrs["state"] = "complete"
        self._f.flush()
        self._f.close()
        self._f = None

        descriptor = os.open(self._tmp_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(self._tmp_path, self._final_path)
        parent_descriptor = os.open(self._final_path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    @property
    def name(self) -> str:
        return self._final_path.stem


class h5EpochReaderV23:
    """Read v2.3 NumPy windows directly from HDF5."""

    def __init__(self, path, raw_chunk_cache_bytes: int = 8 * 2**20):
        self.path = str(path)
        self._f = h5py.File(
            self.path, "r", rdcc_nbytes=int(raw_chunk_cache_bytes)
        )
        if self._f.attrs.get("version") != "2.3":
            self.close()
            raise ValueError(f"Expected HDF5 v2.3: {self.path}")
        if self._f.attrs.get("state") != "complete":
            self.close()
            raise ValueError(f"HDF5 materialisation is incomplete: {self.path}")
        self.ch_names = self._f["channels/name"].asstr()[:].tolist()
        self._channel_lookup = {
            name: index for index, name in enumerate(self.ch_names)
        }

    def read_epoch(self, index: int, channels=None, start=None, stop=None):
        start = 0 if start is None else int(start)
        stop = self._f["data"].shape[2] if stop is None else int(stop)
        if channels is None:
            return np.asarray(
                self._f["data"][index, :, start:stop], dtype="float32"
            )
        requested = np.asarray(
            [
                self._channel_lookup[name] if isinstance(name, str) else int(name)
                for name in channels
            ],
            dtype="int64",
        )
        order = np.argsort(requested)
        result = np.asarray(
            self._f[
                "data"
            ][index, requested[order].tolist(), start:stop],
            dtype="float32",
        )
        return result[np.argsort(order)]

    def resolve_source(self, source_id: int) -> dict:
        source_id = int(source_id)
        return {
            "source_id": source_id,
            "source_uid": self._f["sources/uid"].asstr()[source_id],
            "source_name": self._f["sources/name"].asstr()[source_id],
            "file_path": self._f["sources/file_path"].asstr()[source_id],
        }

    def source_id_for_epoch(self, index: int) -> int:
        return int(self._f["epochs/source_id"][index])

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
