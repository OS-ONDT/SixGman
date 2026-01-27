import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

import sixdman
print("DEBUG: Imported sixdman version:", getattr(sixdman, "__version__", "unknown"))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SixDman'
copyright = '2026, Matin Rafiei Forooshani'
author = 'Matin Rafiei Forooshani'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # Google/NumPy style docstrings
    'sphinx_autodoc_typehints',  # Show type hints
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'


html_theme_options = {
    "repository_url": "https://github.com/UC3M-ONDT/SixDman",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "use_issues_button": True,
}


autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}