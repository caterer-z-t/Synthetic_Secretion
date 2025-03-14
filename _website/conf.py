# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Synthetic Secretion'
copyright = '2025, Zachary Caterer'
author = 'Zachary Caterer'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- Extensions ---------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",         # Automatically document Python modules
    "sphinx.ext.napoleon",        # Support for Google style docstrings
    "sphinx_copybutton",          # Adds a "copy" button to code blocks
    # "m2r2",                     # Support for Markdown files
    "nbsphinx",                   # Support for Jupyter Notebooks
    "myst_parser",                # Support for markdown files using
]

source_suffix = {
    ".rst": "restructuredtext",
    # ".txt": "markdown",
    ".md": "markdown",
}
html_js_files = [
    "readthedocs.js",
]

templates_path = ["_templates"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Define colors for light and dark modes
cellphenox_dark_blue = "#317EC2"
cellphenox_light_blue = "#B9DBF4"
cellphenox_pink = "#F3CDCC"
cellphenox_red = "#C25757"

# Furo theme option colors specified here
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": cellphenox_red,
        "color-brand-content": cellphenox_red,
        "color-api-pre-name": cellphenox_red,
        "color-api-name": cellphenox_red,
    },
    "dark_css_variables": {
        "color-brand-primary": cellphenox_dark_blue,  # Use dark blue for primary
        "color-brand-content": cellphenox_dark_blue,  # Use dark blue for content
        "color-api-pre-name": cellphenox_dark_blue,  # Use dark blue for API names
        "color-api-name": cellphenox_dark_blue,  # Use dark blue for API pre-names
    },
    "sidebar_hide_name": False,
}

# Path to the logo file
# html_logo = "../logo/pycpx.svg"
