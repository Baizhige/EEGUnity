from eegunity.modules.parser.eeg_parser import channel_name_parser


def test_legacy_dual_prefixes_are_canonical_eeg():
    assert channel_name_parser("Dual:T8-P8") == "eeg:T8-P8"
    assert channel_name_parser("EEGDual:T8-P8") == "eeg:T8-P8"
    assert channel_name_parser("eeg:T8-P8") == "eeg:T8-P8"


def test_legacy_dual_prefix_does_not_create_new_channel_type():
    parsed = channel_name_parser("Dual:Fp1-F7, EEGDual:F7-T7")
    assert parsed == "eeg:Fp1-F7, eeg:F7-T7"
    assert "dual:" not in parsed.casefold()
