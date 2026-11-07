import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from eegunity.modules.batch.eeg_batch import _format_parsed_channels
from eegunity.unifieddataset import UnifiedDataset


def _raw(sfreq=100.0, n_times=100):
    info = mne.create_info(["C3", "C4"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    data = np.vstack(
        [
            np.sin(np.linspace(0, 8 * np.pi, n_times)),
            np.cos(np.linspace(0, 6 * np.pi, n_times)),
        ]
    )
    return mne.io.RawArray(data, info, verbose=False)


def _dataset_from_raws(directory, raws, domain="demo"):
    directory.mkdir(parents=True, exist_ok=True)
    for index, raw in enumerate(raws):
        raw.save(directory / f"rec{index}_raw.fif", overwrite=True, verbose=False)
    dataset = UnifiedDataset(
        dataset_path=str(directory),
        domain_tag=domain,
        is_unzip=False,
    )
    locator = dataset.get_locator().copy()
    locator.index = [10 + index * 7 for index in range(len(locator))]
    locator["Note"] = [f"note-{index}" for index in range(len(locator))]
    dataset.set_locator(locator)
    return dataset


def _set_typed_channel_names(dataset, channel_names="eeg:C3, eeg:C4"):
    """Replace locator channel names with an explicit EEGUnity type schema."""
    locator = dataset.get_locator().copy()
    locator["Channel Names"] = channel_names
    dataset.set_locator(locator)
    return channel_names


def _file_hash(path):
    """Return a SHA-256 hash for a regression-test source file."""
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _mark_locator_source(dataset):
    """Set stale locator state that a derived commit must clear."""
    dataset.set_shared_attr({"locator_path": "old_locator.csv"})


class TestLocatorMetadata(unittest.TestCase):
    def test_format_type_selects_typed_or_plain_names(self):
        self.assertEqual(_format_parsed_channels("eeg:C3, eeg:Cz", "EEGUnity"), "eeg:C3, eeg:Cz")
        self.assertEqual(_format_parsed_channels("eeg:C3, stim:marker", "normal"), "C3, marker")
        with self.assertRaises(ValueError):
            _format_parsed_channels("eeg:C3", "other")

    def test_domain_mean_uses_row_position_when_index_is_not_positional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(), _raw()])
            dataset.eeg_batch.process_mean_std(domain_mean=True)
            locator = dataset.get_locator()
            self.assertEqual(list(locator.index), [10, 17])
            self.assertEqual(locator["Note"].tolist(), ["note-0", "note-1"])
            self.assertIn("all_eeg", locator["MEAN STD"].iloc[0])
            self.assertEqual(locator["MEAN STD"].iloc[0].keys(), locator["MEAN STD"].iloc[1].keys())

    def test_domain_mean_keeps_failed_rows_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset_from_raws(Path(directory), [_raw(), _raw()])
            loaded = _raw()
            with patch.object(
                dataset.eeg_batch,
                "_get_data_row",
                side_effect=[loaded, RuntimeError("broken recording")],
            ):
                dataset.eeg_batch.process_mean_std(domain_mean=True, miss_bad_data=True)

            values = dataset.get_locator()["MEAN STD"].tolist()
            self.assertIn("all_eeg", values[0])
            self.assertIsNone(values[1])

    def test_format_channel_names_honors_format_type(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset_from_raws(Path(directory), [_raw()])
            dataset.eeg_batch.format_channel_names(format_type="normal")
            self.assertEqual(dataset.get_locator()["Channel Names"].iloc[0], "C3, C4")
            dataset.eeg_batch.format_channel_names(format_type="EEGUnity")
            self.assertEqual(dataset.get_locator()["Channel Names"].iloc[0], "eeg:C3, eeg:C4")

    def test_save_as_other_csv_uses_csv_suffix_and_keeps_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw()])
            output = root / "out"
            output.mkdir()
            _mark_locator_source(dataset)
            dataset.eeg_batch.save_as_other(str(output), format="csv")
            saved = dataset.get_locator()["File Path"].iloc[0]
            self.assertTrue(saved.endswith(".csv"))
            self.assertTrue(Path(saved).is_file())
            frame = pd.read_csv(saved)
            self.assertIn("timestamp", frame.columns)
            self.assertIn("C3", frame.columns)
            locator = dataset.get_locator()
            self.assertEqual(locator["Note"].iloc[0], "note-0")
            self.assertEqual(locator["Channel Names"].iloc[0], "C3, C4, marker")
            self.assertEqual(locator["Number of Channels"].iloc[0], "3")
            self.assertEqual(locator["Data Shape"].iloc[0], "(3, 100)")
            self.assertEqual(dataset.get_shared_attr()["dataset_path"], str(output))
            self.assertIsNone(dataset.get_shared_attr()["locator_path"])
            self.assertEqual(locator["Completeness Check"].iloc[0], "Completed")

    def test_save_as_other_csv_collision_keeps_csv_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw()])
            output = root / "out"
            output.mkdir()
            (output / "rec0_raw_raw.csv").write_text("existing", encoding="utf-8")

            dataset.eeg_batch.save_as_other(str(output), format="csv")

            saved = Path(dataset.get_locator()["File Path"].iloc[0])
            self.assertEqual(saved.suffix, ".csv")
            self.assertNotEqual(saved.name, "rec0_raw_raw.csv")
            self.assertTrue(saved.is_file())

    def test_save_as_other_csv_preserves_typed_locator_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw()])
            _set_typed_channel_names(dataset)
            output = root / "out"
            output.mkdir()

            dataset.eeg_batch.save_as_other(str(output), format="csv")

            locator = dataset.get_locator()
            self.assertEqual(
                locator["Channel Names"].iloc[0],
                "eeg:C3, eeg:C4, stim:marker",
            )
            self.assertEqual(locator["Number of Channels"].iloc[0], "3")
            self.assertEqual(locator["Data Shape"].iloc[0], "(3, 100)")

    def test_explicit_channel_type_disable_does_not_duplicate_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw()])
            _set_typed_channel_names(dataset)
            output = root / "out"
            output.mkdir()

            dataset.eeg_batch.save_as_other(
                str(output),
                format="csv",
                preserve_events=False,
                get_data_row_params={"is_set_channel_type": False},
            )

            self.assertEqual(
                dataset.get_locator()["Channel Names"].iloc[0],
                "eeg:C3, eeg:C4",
            )

    def test_filter_keeps_locator_columns_and_ignores_extra_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(n_times=1000)])
            source_path = dataset.get_locator()["File Path"].iloc[0]
            source_hash = _file_hash(source_path)
            output = root / "filtered"
            output.mkdir()
            _mark_locator_source(dataset)
            decoy = _raw()
            decoy.save(output / "already_there_raw.fif", overwrite=True, verbose=False)
            dataset.eeg_batch.filter(str(output), l_freq=1.0, h_freq=20.0)
            locator = dataset.get_locator()
            self.assertEqual(len(locator), 1)
            self.assertTrue(locator["File Path"].iloc[0].endswith("_filter_raw.fif"))
            self.assertFalse(
                (output / "already_there_raw.fif").samefile(locator["File Path"].iloc[0])
            )
            self.assertEqual(locator["Note"].iloc[0], "note-0")
            self.assertEqual(locator["Domain Tag"].iloc[0], "demo")
            self.assertEqual(_file_hash(source_path), source_hash)
            self.assertEqual(dataset.get_shared_attr()["dataset_path"], str(output))
            self.assertIsNone(dataset.get_shared_attr()["locator_path"])
            self.assertEqual(locator["Completeness Check"].iloc[0], "Completed")

    def test_filter_miss_bad_data_keeps_only_successful_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(
                root / "src",
                [_raw(n_times=1000), _raw(n_times=1000)],
            )
            output = root / "filtered"
            output.mkdir()
            with patch.object(
                dataset.eeg_batch,
                "_get_data_row",
                side_effect=[_raw(n_times=1000), RuntimeError("broken recording")],
            ):
                dataset.eeg_batch.filter(
                    str(output),
                    l_freq=1.0,
                    h_freq=20.0,
                    miss_bad_data=True,
                )

            locator = dataset.get_locator()
            self.assertEqual(len(locator), 1)
            self.assertEqual(locator["Note"].tolist(), ["note-0"])
            self.assertTrue(Path(locator["File Path"].iloc[0]).is_file())

    def test_filter_failure_does_not_commit_locator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(n_times=1000)])
            original = dataset.get_locator().copy(deep=True)
            output = root / "filtered"
            output.mkdir()
            with patch.object(
                dataset.eeg_batch,
                "_get_data_row",
                side_effect=RuntimeError("broken recording"),
            ):
                with self.assertRaises(RuntimeError):
                    dataset.eeg_batch.filter(
                        str(output),
                        l_freq=1.0,
                        h_freq=20.0,
                        miss_bad_data=False,
                    )

            pd.testing.assert_frame_equal(dataset.get_locator(), original)

    def test_filter_preserves_typed_locator_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(n_times=1000)])
            expected = _set_typed_channel_names(dataset)
            output = root / "filtered"
            output.mkdir()

            dataset.eeg_batch.filter(str(output), l_freq=1.0, h_freq=20.0)

            locator = dataset.get_locator()
            self.assertEqual(locator["Channel Names"].iloc[0], expected)
            self.assertEqual(locator["Number of Channels"].iloc[0], "2")

    def test_ica_uses_the_shared_derived_locator_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(n_times=1000)])
            expected = _set_typed_channel_names(dataset)
            _mark_locator_source(dataset)
            output = root / "ica"
            output.mkdir()

            with patch("eegunity.modules.batch.eeg_batch.mne.preprocessing.ICA") as ica_class:
                dataset.eeg_batch.ica(str(output))

            ica_class.return_value.fit.assert_called_once()
            ica_class.return_value.apply.assert_called_once()
            locator = dataset.get_locator()
            self.assertEqual(locator["Channel Names"].iloc[0], expected)
            self.assertEqual(locator["File Type"].iloc[0], "standard_data")
            self.assertEqual(locator["Completeness Check"].iloc[0], "Completed")
            self.assertEqual(dataset.get_shared_attr()["dataset_path"], str(output))
            self.assertIsNone(dataset.get_shared_attr()["locator_path"])

    def test_get_meta_persists_post_load_typed_channel_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset_from_raws(Path(directory), [_raw()])

            dataset.eeg_batch.get_meta()

            locator = dataset.get_locator()
            self.assertEqual(locator["Channel Names"].iloc[0], "eeg:C3, eeg:C4")
            self.assertEqual(locator["Channel Types"].iloc[0], "eeg, eeg")

    def test_resample_preserves_duration_and_locator_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(sfreq=100.0, n_times=200)])
            before = float(dataset.get_locator()["Duration"].iloc[0])
            output = root / "resampled"
            output.mkdir()
            _mark_locator_source(dataset)
            dataset.eeg_batch.resample(str(output), resample_params={"sfreq": 50})
            locator = dataset.get_locator()
            after = float(locator["Duration"].iloc[0])
            self.assertAlmostEqual(after, before, delta=0.05)
            self.assertAlmostEqual(float(locator["Sampling Rate"].iloc[0]), 50.0)
            self.assertEqual(locator["Note"].iloc[0], "note-0")
            self.assertEqual(len(locator), 1)
            self.assertTrue(locator["File Path"].iloc[0].endswith("_resampled_raw.fif"))
            self.assertEqual(dataset.get_shared_attr()["dataset_path"], str(output))
            self.assertIsNone(dataset.get_shared_attr()["locator_path"])
            self.assertEqual(locator["Completeness Check"].iloc[0], "Completed")

    def test_resample_preserves_typed_locator_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(sfreq=100.0, n_times=200)])
            expected = _set_typed_channel_names(dataset)
            output = root / "resampled"
            output.mkdir()

            dataset.eeg_batch.resample(str(output), resample_params={"sfreq": 50})

            locator = dataset.get_locator()
            self.assertEqual(locator["Channel Names"].iloc[0], expected)
            self.assertEqual(locator["Number of Channels"].iloc[0], "2")
            self.assertEqual(locator["Data Shape"].iloc[0], "(2, 100)")


if __name__ == "__main__":
    unittest.main()
