import unittest
from unittest.mock import MagicMock, patch

from eegunity.modules.batch.eeg_scores_modified_mne import (
    compute_quality_score_mne,
)


def _mock_raw():
    raw = MagicMock()
    raw.load_data.return_value = raw
    raw.filter.return_value = raw
    raw.ch_names = ["C3", "EOG-left", "ECG-main"]
    raw.info = {
        "ch_names": raw.ch_names,
        "chs": [{"kind": 2}, {"kind": 2}, {"kind": 2}],
    }
    return raw


class TestQualityScore(unittest.TestCase):
    @patch("eegunity.modules.batch.eeg_scores_modified_mne.mne.pick_types")
    @patch("eegunity.modules.batch.eeg_scores_modified_mne.mne.preprocessing.ICA")
    def test_named_artifact_channels_are_passed_to_mne(self, ica_class, pick_types):
        pick_types.return_value = []
        ica = ica_class.return_value
        ica.find_bads_eog.return_value = ([0], [0.5])
        ica.find_bads_muscle.return_value = ([], [])
        ica.find_bads_ecg.return_value = ([1], [0.4])

        score = compute_quality_score_mne(_mock_raw())

        ica.find_bads_eog.assert_called_once_with(
            unittest.mock.ANY,
            ch_name=["EOG-left"],
        )
        ica.find_bads_ecg.assert_called_once_with(
            unittest.mock.ANY,
            ch_name=["ECG-main"],
        )
        self.assertEqual(score, "0.33333333333333337, 1, 0, 1")

    @patch("eegunity.modules.batch.eeg_scores_modified_mne.mne.pick_types")
    @patch("eegunity.modules.batch.eeg_scores_modified_mne.mne.preprocessing.ICA")
    def test_eog_detection_failure_does_not_drop_score(self, ica_class, pick_types):
        pick_types.return_value = []
        ica = ica_class.return_value
        ica.find_bads_eog.side_effect = RuntimeError("no usable EOG")
        ica.find_bads_muscle.return_value = ([], [])
        ica.find_bads_ecg.return_value = ([], [])

        score = compute_quality_score_mne(_mock_raw())

        self.assertEqual(score, "1.0, 0, 0, 0")


if __name__ == "__main__":
    unittest.main()
