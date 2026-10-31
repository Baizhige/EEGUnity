import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eegunity.modules.parser.eeg_parser import process_mne_files
from eegunity.unifieddataset import UnifiedDataset
from eegunity.utils.parallel import parallel_execute


def _raw():
    info = mne.create_info(["C3"], sfreq=100.0, ch_types=["eeg"])
    data = np.sin(np.linspace(0, 4 * np.pi, 100))[None, :]
    return mne.io.RawArray(data, info, verbose=False)


class TestScanAndProcessPool(unittest.TestCase):
    def test_process_mode_unpickles_a_closure_once_per_worker(self):
        offset = 4

        def app(row):
            return row["x"] + offset

        results = parallel_execute(
            tasks=[(0, {"x": 1}, True), (1, {"x": 2}, False), (2, {"x": 3}, True)],
            app_func=app,
            is_patch=False,
            result_type="value",
            execution_mode="process",
            num_workers=1,
        )
        self.assertEqual(results, [5, None, 7])

    def test_directory_scan_does_not_open_non_signal_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root / "readme.json"
            notes.write_text('{"not": "eeg"}', encoding="utf-8")
            raw_path = root / "rec_raw.fif"
            _raw().save(raw_path, overwrite=True, verbose=False)
            locator = pd.DataFrame({
                "File Path": [str(notes), str(raw_path)],
                "Domain Tag": ["demo", "demo"],
                "File Type": ["", ""],
                "Data Shape": ["", ""],
                "Channel Names": ["", ""],
                "Number of Channels": ["", ""],
                "Sampling Rate": ["", ""],
                "Duration": ["", ""],
                "Completeness Check": ["", ""],
            })
            updated = process_mne_files(locator, verbose="CRITICAL", num_workers=0)
            self.assertEqual(updated.loc[0, "File Type"], "unknown")
            self.assertEqual(updated.loc[0, "Error"], "")
            self.assertEqual(updated.loc[1, "File Type"], "standard_data")

    def test_sample_filter_and_replace_paths_keep_the_locator_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src"
            source.mkdir()
            _raw().save(source / "rec_raw.fif", overwrite=True, verbose=False)
            dataset = UnifiedDataset(
                dataset_path=str(source),
                domain_tag="demo",
                is_unzip=False,
            )
            locator = dataset.get_locator().copy()
            locator.index = [8]
            locator["Note"] = "keep"
            dataset.set_locator(locator)

            dataset.eeg_batch.sample_filter(sampling_rate=(50, 150))
            kept = dataset.get_locator()
            self.assertEqual(list(kept.index), [8])
            self.assertEqual(kept["Note"].iloc[0], "keep")

            dataset.eeg_batch.sample_filter(sampling_rate=(1000, 2000))
            self.assertEqual(len(dataset.get_locator()), 0)

            dataset.set_locator(locator)
            dataset.eeg_batch.replace_paths(str(source), str(root / "moved"))
            moved = dataset.get_locator()["File Path"].iloc[0]
            self.assertTrue(moved.startswith(str(root / "moved")))
            self.assertEqual(dataset.get_locator().index.tolist(), [8])
            self.assertEqual(dataset.get_locator()["Note"].iloc[0], "keep")


if __name__ == "__main__":
    unittest.main()
