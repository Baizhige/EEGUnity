import json
import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from eegunity.utils.h5_v23 import h5EpochDatasetV23, h5EpochReaderV23


class TestHDF5V23(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, np.ndarray]:
        rng = np.random.default_rng(42)
        first = rng.normal(size=(3, 4, 2400)).astype("float32")
        second = rng.normal(size=(2, 4, 2400)).astype("float32")
        third = rng.normal(size=(1, 4, 2400)).astype("float32")
        writer = h5EpochDatasetV23(
            root,
            "fixture",
            root_attrs={
                "dataset_id": "fixture-dataset",
                "pipeline_fingerprint": "pipeline-test",
            },
        )
        common = {
            "sfreq": 200.0,
            "ch_names": ["C3", "C4", "P3", "P4"],
            "ch_types": ["eeg"] * 4,
            "info_bytes": pickle.dumps({"sfreq": 200.0}),
        }
        writer.add_epochs(
            group_name="recording",
            event_name="left",
            epoch_data=first,
            source_attrs={
                "file_path": "/raw/site-a/recording.edf",
                "age": "21",
                "sex": "F",
            },
            misc_values={"score": np.asarray([0.1, 0.2, 0.3])},
            epoch_meta={"run": ["1", "1", "1"]},
            channel_mask=np.asarray([1, 1, 1, 0], dtype="uint8"),
            source_start_samples=np.asarray([0, 800, 1600]),
            **common,
        )
        writer.add_epochs(
            group_name="recording",
            event_name="right",
            epoch_data=second,
            source_attrs={
                "file_path": "/raw/site-a/recording.edf",
                "age": "21",
                "sex": "F",
            },
            epoch_meta={"run": "2"},
            channel_mask=np.asarray([1, 1, 1, 0], dtype="uint8"),
            source_start_samples=np.asarray([2400, 3200]),
            **common,
        )
        writer.add_epochs(
            group_name="recording",
            event_name="left",
            epoch_data=third,
            source_attrs={
                "file_path": "/raw/site-b/recording.edf",
                "age": "29",
                "sex": "M",
            },
            misc_values={"score": np.asarray([0.8])},
            epoch_meta={"run": "1"},
            channel_mask=np.ones(4, dtype="uint8"),
            source_start_samples=np.asarray([0]),
            **common,
        )
        writer.save()
        return root / "fixture.hdf5", np.concatenate([first, second, third])

    def test_schema_source_identity_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self._write_fixture(Path(directory))
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["version"], "2.3")
                self.assertEqual(handle.attrs["state"], "complete")
                self.assertEqual(handle["data"].shape, expected.shape)
                self.assertEqual(handle["data"].chunks, (1, 4, 800))
                self.assertTrue(handle["data"].shuffle)
                self.assertEqual(handle["data"].compression, "gzip")
                np.testing.assert_array_equal(
                    handle["epochs/source_id"][:],
                    np.asarray([0, 0, 0, 0, 0, 1], dtype="uint32"),
                )
                np.testing.assert_array_equal(
                    handle["epochs/source_epoch_index"][:],
                    np.asarray([0, 1, 2, 3, 4, 0], dtype="uint32"),
                )
                np.testing.assert_array_equal(
                    handle["epochs/source_start_sample"][:],
                    np.asarray([0, 800, 1600, 2400, 3200, 0]),
                )
                paths = handle["sources/file_path"].asstr()[:].tolist()
                self.assertEqual(
                    paths,
                    ["/raw/site-a/recording.edf", "/raw/site-b/recording.edf"],
                )
                uids = handle["sources/uid"].asstr()[:].tolist()
                self.assertEqual(len(set(uids)), 2)
                self.assertNotIn("source_group", handle["epochs"])
                self.assertEqual(
                    handle["sources/info/age"].asstr()[:].tolist(), ["21", "29"]
                )
                self.assertEqual(
                    handle["sources/info/sex"].asstr()[:].tolist(), ["F", "M"]
                )
                score = handle["epochs/misc/score"][:]
                self.assertTrue(np.isnan(score[3:5]).all())
                self.assertAlmostEqual(float(score[5]), 0.8, places=6)
                self.assertEqual(
                    json.loads(handle.attrs["label_map"]),
                    {"0": "left", "1": "right"},
                )

    def test_reader_direct_window_and_channel_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self._write_fixture(Path(directory))
            with h5EpochReaderV23(path) as reader:
                actual = reader.read_epoch(
                    0, channels=["P4", "C3"], start=800, stop=1600
                )
                np.testing.assert_allclose(
                    actual, expected[0, [3, 0], 800:1600]
                )
                source = reader.resolve_source(reader.source_id_for_epoch(5))
                self.assertEqual(source["source_id"], 1)
                self.assertEqual(
                    source["file_path"], "/raw/site-b/recording.edf"
                )

    def test_non_finite_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = h5EpochDatasetV23(Path(directory), "invalid")
            data = np.ones((1, 2, 800), dtype="float32")
            data[0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                writer.add_epochs(
                    group_name="source",
                    event_name="event",
                    epoch_data=data,
                    info_bytes=b"",
                    source_attrs={"file_path": "/raw/source.edf"},
                    sfreq=200.0,
                    ch_names=["C3", "C4"],
                    ch_types=["eeg", "eeg"],
                )


if __name__ == "__main__":
    unittest.main()
