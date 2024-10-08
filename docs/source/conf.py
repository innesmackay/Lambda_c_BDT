import sys
import os

sys.path.insert(0, os.path.abspath('../../src/bdt/apps'))
sys.path.insert(0, os.path.abspath('../../src/bdt/utils'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Lambda_c_BDT'
copyright = '2024, Innes Mackay'
author = 'Innes Mackay'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',	     # To generate autodocs
    'sphinx.ext.mathjax',           # autodoc with maths
    'sphinx.ext.napoleon'           # For auto-doc configuration
]

#napoleon_google_docstring = False   # Turn off googledoc strings
#napoleon_numpy_docstring = True     # Turn on numpydoc strings
#napoleon_use_ivar = True 	     # For maths symbology

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
