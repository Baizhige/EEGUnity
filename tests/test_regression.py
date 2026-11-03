import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest

from eegunity.modules.parser.eeg_parser import (
    convert_unit,
    get_data_row,
    infer_channel_unit,
)
from eegunity.unifieddataset import UnifiedDataset
from eegunity.utils.label_channel import (
    is_misc_channel,
    misc_task_name,
    resample_raw_with_labels,
)


def _raw(names, types, data, sfreq=100.0):
    info = mne.create_info(names, sfreq=sfreq, ch_types=types)
    return mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)


def test_infer_channel_unit_uses_mean_amplitude():
    samples = np.ones(8)
    assert infer_channel_unit("C3", samples * 20, "eeg") == "uV"
    assert infer_channel_unit("C3", samples * 0.01, "eeg") == "mV"
    assert infer_channel_unit("C3", samples * 1e-6, "eeg") == "V"


def test_convert_unit_scales_microvolts_to_volts():
    raw = _raw(["C3"], ["eeg"], [[2.0, 4.0]])
    raw.info["chs"][0]["eegunity_unit"] = "uV"

    converted = convert_unit(raw, "V")

    np.testing.assert_allclose(converted.get_data()[0], [2e-6, 4e-6])
    assert converted.info["chs"][0]["eegunity_unit_converted"] == "V"


def test_convert_unit_rejects_an_unknown_unit():
    raw = _raw(["C3"], ["eeg"], [[1.0, 1.0]])
    with pytest.raises(ValueError, match="Invalid unit"):
        convert_unit(raw, "nV")


def test_misc_label_names():
    assert is_misc_channel("misc:reaction_time")
    assert not is_misc_channel("eeg:C3")
    assert misc_task_name("misc:reaction_time") == "reaction_time"
    with pytest.raises(ValueError, match="Not a misc"):
        misc_task_name("eeg:C3")


def test_resample_keeps_misc_labels_on_their_original_values():
    eeg = np.linspace(0.0, 1.0, 100)
    label = np.array([1.0] * 50 + [2.0] * 50)
    raw = _raw(
        ["C3", "misc:reaction_time"],
        ["eeg", "misc"],
        np.vstack([eeg, label]),
    )

    resampled = resample_raw_with_labels(raw, sfreq=50)
    kept = resampled.get_data()[1]

    assert resampled.n_times == 50
    assert set(np.unique(kept)).issubset({1.0, 2.0})
    assert kept[0] == 1.0
    assert kept[-1] == 2.0


def test_saved_fif_round_trips_through_the_locator():
    raw = _raw(["C3", "C4"], ["eeg", "eeg"], np.ones((2, 256)), sfreq=256.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo_raw.fif"
        raw.save(path, overwrite=True, verbose=False)
        dataset = UnifiedDataset(
            dataset_path=tmp,
            domain_tag="demo",
            verbose="CRITICAL",
            num_workers=0,
            is_unzip=False,
        )
        locator = dataset.get_locator()
        assert len(locator) == 1
        row = locator.iloc[0].to_dict()
        assert str(row["File Path"]).endswith("demo_raw.fif")

        loaded = get_data_row(row, preload=True, is_set_montage=False)
        assert loaded.info["sfreq"] == 256.0
        assert loaded.n_times == 256
        assert any("C3" in name for name in loaded.ch_names)
        assert any("C4" in name for name in loaded.ch_names)
