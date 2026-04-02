import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

version_ns = {}
exec((ROOT / "eegunity" / "_version.py").read_text(encoding="utf-8"), version_ns)

# -- Project information -----------------------------------------------------
project = "EEGUnity"
copyright = "2026, EEGUnity Team"
author = "EEGUnity Team"
version = version_ns["__version__"]
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
version_match = "latest"
html_theme_options = {
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": version_match,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
}
html_logo = "_static/logo.png"
html_static_path = ["_static"]
