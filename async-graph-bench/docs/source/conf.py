# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("..src.async_graph_bench"))

project = 'async_graph_benchmarking'
copyright = '2025, Philip Müller'
author = 'Philip Müller'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_external_toc",

    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
    "sphinx_autodoc_typehints",  # type hints in docs

    # "autoapi.extension"
]

# autoapi_dirs = ["../your_package"]

# Ensure Sphinx looks for the file named exactly below
external_toc_path = "_toc.yml"  # <- note: .yml, not .yaml (use this unless you set a custom path)

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
