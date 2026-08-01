import json
import tempfile
import unittest
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from eegunity.modules.batch.method_mixin_epoch import (
    EEGBatchMixinEpoch,
    _make_epoch_writer,
)
from eegunity.utils.h5 import h5EpochDatasetV2
from eegunity.utils.h5_v23 import h5EpochDatasetV23


class _SyntheticBatch(EEGBatchMixinEpoch):
    def __init__(self):
        rng = np.random.default_rng(7)
        info = mne.create_info(["C3", "C4"], sfreq=200.0, ch_types=["eeg", "eeg"])
        self.raw = mne.io.RawArray(
            rng.normal(size=(2, 1200)),
            info,
            verbose=False,
        )
        self.raw.info["description"] = json.dumps(
            {
                "eegunity_description": {
                    "age": 31,
                    "sex": "F",
                    "site": "pilot-site",
                }
            }
        )
        self.locator = pd.DataFrame(
            [
                {
                    "File Path": "/raw/site-a/recording.edf",
                    "Completeness Check": "Completed",
                    "Number of Channels": 2,
                }
            ]
        )

    def get_shared_attr(self):
        return {"locator": self.locator, "num_workers": 0}

    def _get_data_row(self, row, **kwargs):
        return self.raw.copy()

    def batch_process(
        self,
        con_func,
        app_func,
        is_patch,
        result_type=None,
        execution_mode=None,
    ):
        for _, row in self.locator.iterrows():
            if con_func(row):
                app_func(row)
        return None


class TestHDF5V23BatchIntegration(unittest.TestCase):
    def test_default_segmentation_is_v23_and_has_exact_sample_count(self):
        batch = _SyntheticBatch()
        with tempfile.TemporaryDirectory() as directory:
            batch.epoch_by_segmentation_hdf5(
                directory,
                file_name_prefix="segments",
                exclude_bad=False,
                segment_params={"segment_length": 1.0, "overlap": 0.0},
                epoch_params={"baseline": None, "preload": True, "verbose": False},
                root_attrs={
                    "dataset_id": "fixture-dataset",
                    "pipeline_fingerprint": "pipeline-sha256",
                },
            )
            path = Path(directory) / "segments.hdf5"
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["version"], "2.3")
                self.assertEqual(handle.attrs["state"], "complete")
                self.assertEqual(handle.attrs["dataset_id"], "fixture-dataset")
                self.assertEqual(handle.attrs["pipeline_fingerprint"], "pipeline-sha256")
                self.assertEqual(handle["data"].shape, (6, 2, 200))
                np.testing.assert_array_equal(
                    handle["epochs/source_start_sample"][:],
                    np.arange(0, 1200, 200),
                )
                self.assertEqual(
                    handle["channels/name"].asstr()[:].tolist(),
                    ["C3", "C4"],
                )
                self.assertEqual(
                    handle["channels/type"].asstr()[:].tolist(),
                    ["eeg", "eeg"],
                )
                self.assertEqual(
                    handle["sources/file_path"].asstr()[0],
                    "/raw/site-a/recording.edf",
                )
                self.assertEqual(
                    handle["sources/info/site"].asstr()[0],
                    "pilot-site",
                )

    def test_writer_selection_preserves_explicit_legacy_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            current = _make_epoch_writer("v2.3", directory, "current")
            self.assertIsInstance(current, h5EpochDatasetV23)
            with self.assertWarns(FutureWarning):
                legacy = _make_epoch_writer("v2", directory, "legacy")
            self.assertIsInstance(legacy, h5EpochDatasetV2)


if __name__ == "__main__":
    unittest.main()
