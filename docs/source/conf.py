# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'FDGAN'
copyright = '2025, Ramsses De Los Santos Mendoza'
author = 'Ramsses De Los Santos Mendoza'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc.typehints",
]

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "kornia",
    "numpy",
    "PIL", 
    "tqdm",
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_use_ivar = True


# Autosummary settings
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']