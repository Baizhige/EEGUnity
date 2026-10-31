import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eegunity.modules.batch.eeg_batch import _format_parsed_channels, _shape_pair
from eegunity.unifieddataset import UnifiedDataset


def _raw(sfreq=100.0, n_times=100):
    info = mne.create_info(["C3", "C4"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    data = np.vstack([
        np.sin(np.linspace(0, 8 * np.pi, n_times)),
        np.cos(np.linspace(0, 6 * np.pi, n_times)),
    ])
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


class TestLocatorMetadata(unittest.TestCase):
    def test_shape_pair_rejects_eval_payload(self):
        self.assertEqual(_shape_pair("(2, 100)"), (2, 100))
        with self.assertRaises(ValueError):
            _shape_pair("__import__('os')")

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
            dataset.eeg_batch.save_as_other(str(output), format="csv")
            saved = dataset.get_locator()["File Path"].iloc[0]
            self.assertTrue(saved.endswith(".csv"))
            self.assertTrue(Path(saved).is_file())
            frame = pd.read_csv(saved)
            self.assertIn("timestamp", frame.columns)
            self.assertIn("C3", frame.columns)
            self.assertEqual(dataset.get_locator()["Note"].iloc[0], "note-0")

    def test_filter_keeps_locator_columns_and_ignores_extra_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw()])
            output = root / "filtered"
            output.mkdir()
            decoy = _raw()
            decoy.save(output / "already_there_raw.fif", overwrite=True, verbose=False)
            dataset.eeg_batch.filter(str(output), l_freq=1.0, h_freq=20.0)
            locator = dataset.get_locator()
            self.assertEqual(len(locator), 1)
            self.assertTrue(locator["File Path"].iloc[0].endswith("_filter.fif"))
            self.assertFalse((output / "already_there_raw.fif").samefile(locator["File Path"].iloc[0]))
            self.assertEqual(locator["Note"].iloc[0], "note-0")
            self.assertEqual(locator["Domain Tag"].iloc[0], "demo")

    def test_resample_preserves_duration_and_locator_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset_from_raws(root / "src", [_raw(sfreq=100.0, n_times=200)])
            before = float(dataset.get_locator()["Duration"].iloc[0])
            output = root / "resampled"
            output.mkdir()
            dataset.eeg_batch.resample(str(output), resample_params={"sfreq": 50})
            locator = dataset.get_locator()
            after = float(locator["Duration"].iloc[0])
            self.assertAlmostEqual(after, before, delta=0.05)
            self.assertAlmostEqual(float(locator["Sampling Rate"].iloc[0]), 50.0)
            self.assertEqual(locator["Note"].iloc[0], "note-0")
            self.assertEqual(len(locator), 1)
            self.assertTrue(locator["File Path"].iloc[0].endswith("_resampled_raw.fif"))


if __name__ == "__main__":
    unittest.main()
