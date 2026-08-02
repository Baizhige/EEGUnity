import mne
import numpy as np
import pytest

from eegunity.utils.normalize import normalize_mne


def _raw(data):
    info = mne.create_info(
        ["varying", "constant", "label", "trigger"],
        sfreq=100.0,
        ch_types=["eeg", "eeg", "misc", "stim"],
    )
    return mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)


def test_normalize_maps_constant_signal_to_zero_and_preserves_labels():
    source = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [7.0, 7.0, 7.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
            [0.0, 1.0, 0.0, 2.0],
        ]
    )
    raw = _raw(source)

    result = normalize_mne(raw)
    actual = result.get_data()

    assert result is raw
    np.testing.assert_allclose(actual[0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(actual[0].std(), 1.0, atol=1e-12)
    np.testing.assert_array_equal(actual[1], np.zeros(4))
    np.testing.assert_array_equal(actual[2:], source[2:])
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_normalize_rejects_non_finite_signal_values(invalid):
    source = np.array(
        [
            [1.0, 2.0, invalid, 4.0],
            [7.0, 7.0, 7.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
            [0.0, 1.0, 0.0, 2.0],
        ]
    )

    with pytest.raises(ValueError, match="varying"):
        normalize_mne(_raw(source))
