#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

master_doc = "index"

project = "spectral_connectivity"
copyright = "2022, Eric L. Denovellis"
author = "Eric L. Denovellis"

# The short X.Y version.
version = "1..0"
# The full version, including alpha/beta/rc tags.
release = "1.0.4"

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "nbsphinx",  # Integrate Jupyter Notebooks and Sphinx
    "numpydoc",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",  # syntax highlighting
]
autosummary_generate = True
add_module_names = False
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = {".rst": "restructuredtext", ".myst": "myst-nb", ".ipynb": "myst-nb"}

language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "setup.py",
    "README.md",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "spectral_connectivitydoc"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
}

# -- MyST and MyST-NB ---------------------------------------------------

# MyST
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

# MyST-NB
nb_execution_mode = "cache"
nb_execution_mode = "off"

# -- Get Jupyter Notebooks ---------------------------------------------------
def copy_tree(src, tar):
    """Copies over notebooks into the documentation folder, so get around an issue where nbsphinx
    requires notebooks to be in the same folder as the documentation folder
    """
    if os.path.exists(tar):
        shutil.rmtree(tar)
    shutil.copytree(src, tar)


copy_tree("../examples", "./examples")
