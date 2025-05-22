import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'EEGUnity'
copyright = '2025, Wenlong You'
author = 'Wenlong You'
release = '0.5.5'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# autodoc_default_options = {
#     'exclude-members': '__weakref__'
# }
#
# templates_path = ['_templates']
# exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
version = "latest"
html_theme_options = {
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": version,

    },
    "navbar_start": ["navbar-logo", "version-switcher"]
}
html_logo = "_static/logo.png"
html_static_path = ['_static']
