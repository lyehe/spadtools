"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Adjust this path to point to the directory containing your spadtools package
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../spadtools/"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SPADTools"
copyright = "2024, Yehe Liu"
author = "Yehe Liu"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# Comment out or remove this line if you're not using any static files
# html_static_path = ["_static"]
