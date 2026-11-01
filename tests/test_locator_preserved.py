import numpy as np
import mne

from eegunity import UnifiedDataset


def test_filter_and_resample_keep_added_locator_columns(tmp_path):
    sfreq = 256
    data = np.random.default_rng(0).normal(size=(2, sfreq * 4)) * 1e-6
    info = mne.create_info(["C3", "C4"], sfreq, ["eeg", "eeg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    source = tmp_path / "source"
    source.mkdir()
    raw.save(source / "demo_raw.fif", overwrite=True, verbose="ERROR")

    dataset = UnifiedDataset(str(source), domain_tag="demo", verbose="CRITICAL", num_workers=0)
    dataset.get_locator()["Infer Unit"] = "uV"

    filtered = tmp_path / "filtered"
    filtered.mkdir()
    dataset.eeg_batch.filter(str(filtered), filter_type="bandpass", l_freq=1.0, h_freq=40.0)
    locator = dataset.get_locator()
    assert list(locator["Infer Unit"]) == ["uV"]
    assert str(locator.loc[0, "File Path"]).endswith("_filter.fif")

    resampled = tmp_path / "resampled"
    resampled.mkdir()
    dataset.eeg_batch.resample(str(resampled), resample_params={"sfreq": 128})
    locator = dataset.get_locator()
    assert list(locator["Infer Unit"]) == ["uV"]
    assert str(locator.loc[0, "Sampling Rate"]) == "128"
    assert str(locator.loc[0, "File Path"]).endswith("_resampled_raw.fif")
