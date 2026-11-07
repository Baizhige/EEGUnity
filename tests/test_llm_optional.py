import sys
from unittest.mock import patch

import pytest

from eegunity.modules.llm_booster.eeg_llm_des_parser import llm_description_file_parser
from eegunity.modules.llm_booster.eeg_llm_file_parser import llm_boost_parser


def test_importing_unifieddataset_does_not_import_llm_extras():
    blocked = ("openai", "pdfplumber")
    for name in list(sys.modules):
        if name in blocked or name.startswith(tuple(item + "." for item in blocked)):
            del sys.modules[name]
    import eegunity.unifieddataset  # noqa: F401

    assert "openai" not in sys.modules
    assert "pdfplumber" not in sys.modules


def test_llm_file_parser_reports_missing_openai(tmp_path):
    csv_path = tmp_path / "eeg.csv"
    csv_path.write_text("t,c3\n0,1\n0.01,2\n", encoding="utf-8")
    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ImportError, match=r"eegunity\[llm\]"):
            llm_boost_parser(str(csv_path), "OpenAI", {}, {})


def test_pdf_description_reports_missing_pdfplumber(tmp_path):
    (tmp_path / "note.pdf").write_bytes(b"%PDF-1.4\n")
    with patch.dict(sys.modules, {"pdfplumber": None}):
        with pytest.raises(ImportError, match=r"eegunity\[llm\]"):
            llm_description_file_parser(str(tmp_path), "OpenAI", {}, {})
