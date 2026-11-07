import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLM_MODULES = [
    ROOT / "eegunity/modules/llm_booster/eeg_llm_booster.py",
    ROOT / "eegunity/modules/llm_booster/eeg_llm_des_parser.py",
    ROOT / "eegunity/modules/llm_booster/eeg_llm_file_parser.py",
]


def _module_level_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    return found


class TestOptionalLlmExtra(unittest.TestCase):
    def test_llm_dependencies_are_not_imported_at_module_scope(self):
        for path in LLM_MODULES:
            imported = _module_level_imports(path)
            self.assertNotIn("openai", imported, path.name)
            self.assertNotIn("pdfplumber", imported, path.name)

    def test_package_metadata_keeps_llm_packages_optional(self):
        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        requires = setup_text.split("install_requires=", 1)[1].split("],", 1)[0]
        extras = setup_text.split("extras_require=", 1)[1]
        self.assertNotIn("openai", requires)
        self.assertNotIn("pdfplumber", requires)
        self.assertIn("openai", extras)
        self.assertIn("pdfplumber", extras)


if __name__ == "__main__":
    unittest.main()
