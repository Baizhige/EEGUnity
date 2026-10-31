import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

from eegunity.modules.batch.in_memory import apply_raw_steps
from eegunity.unifieddataset import UnifiedDataset


def _raw():
    info = mne.create_info(["C3", "C4"], sfreq=100.0, ch_types=["eeg", "eeg"])
    data = np.vstack([
        np.sin(np.linspace(0, 20 * np.pi, 400)),
        np.cos(np.linspace(0, 12 * np.pi, 400)),
    ])
    return mne.io.RawArray(data, info, verbose=False)


class TestInMemorySteps(unittest.TestCase):
    def test_apply_raw_steps_keeps_inplace_mutation_when_step_returns_none(self):
        raw = _raw()

        def scale(recording):
            recording._data *= 2

        updated = apply_raw_steps(raw, [scale])
        self.assertIs(updated, raw)
        self.assertGreater(np.max(np.abs(updated.get_data())), 1.5)

    def test_process_steps_writes_one_file_and_keeps_locator_columns(self):
        calls = []

        def bandpass(recording):
            calls.append("bandpass")
            return recording.filter(l_freq=1.0, h_freq=30.0, verbose=False)

        def slower(recording):
            calls.append("resample")
            return recording.resample(50, verbose=False)

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
            locator["Note"] = "keep"
            before = float(locator["Duration"].iloc[0])
            dataset.set_locator(locator)

            output = root / "out"
            dataset.eeg_batch.process_steps(str(output), [bandpass, slower])

            written = list(output.glob("*.fif"))
            saved = dataset.get_locator()
            self.assertEqual(calls, ["bandpass", "resample"])
            self.assertEqual(len(written), 1)
            self.assertEqual(saved["File Path"].iloc[0], str(written[0]))
            self.assertAlmostEqual(float(saved["Sampling Rate"].iloc[0]), 50.0)
            self.assertAlmostEqual(float(saved["Duration"].iloc[0]), before, delta=0.05)
            self.assertEqual(saved["Note"].iloc[0], "keep")
            self.assertEqual(saved["Domain Tag"].iloc[0], "demo")


if __name__ == "__main__":
    unittest.main()
