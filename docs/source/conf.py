# Configuration file for the Sphinx documentation builder.
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pycsou-GSP'
copyright = '2023, Kaan Okumus'
author = 'Kaan Okumus'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx_togglebutton',
]

templates_path = ['_templates']
exclude_patterns = []
master_doc = "index"
pygments_style = "sphinx"
add_module_names = False
plot_include_source = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': -1,
    'titles_only': False
}
#html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
# -- Options for autosummary extension ---------------------------------------
autosummary_generate = False

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_flags = [
    "members",
    # 'inherited-members',
    "show-inheritance",
]
autodoc_inherit_docstrings = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "NumPy [latest]": ("https://docs.scipy.org/doc/numpy/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/reference", None),
    "dask [latest]": ("https://docs.dask.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/", None)
}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
