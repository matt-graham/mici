"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import collections

project = "Mici"
copyright = "2023, Matt Graham"  # noqa: A001
author = "Matt Graham"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

napoleon_preprocess_types = True
autoclass_content = "both"
python_use_unqualified_type_names = True

autodoc_typehints = "description"

autodoc_default_options = {"inherited-members": True, "special-members": "__call__"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/latest/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "pystan": ("https://pystan.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static", "../images"]

html_theme_options = {
    "logo": {
        "image_light": "../images/mici-logo-rectangular.svg",
        "image_dark": "../images/mici-logo-rectangular-light-text.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/matt-graham/mici",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}


# -- Post process ------------------------------------------------------------

# Remove duplicated NamedTuple attribute docstrings using fix suggested at
# https://stackoverflow.com/a/70459782 and also remove default constructor
# docstring for NamedTuple instances


def _remove_namedtuple_attrib_docstring(_app, _what, _name, obj, skip, _options):
    if isinstance(obj, collections._tuplegetter) or str(obj) in {  # noqa: SLF001
        "<method 'count' of 'tuple' objects>",
        "<method 'index' of 'tuple' objects>",
    }:
        return True
    return skip


def _remove_namedtuple_constructor_lines(_app, _what, _name, obj, _options, lines):
    if (
        isinstance(obj, type)
        and issubclass(obj, tuple)
        and "Create new instance of" in lines[0]
    ):
        lines.clear()


def setup(app):
    app.connect("autodoc-skip-member", _remove_namedtuple_attrib_docstring)
    app.connect("autodoc-process-docstring", _remove_namedtuple_constructor_lines)
