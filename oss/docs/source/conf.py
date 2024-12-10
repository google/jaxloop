# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file for the Sphinx documentation builder."""

# pylint: disable=unused-import,redefined-builtin,g-import-not-at-top

import sys
# Ensure imports work.
from jaxloop import action_loop
from jaxloop import actions
# from jaxloop import eval_loop # Blocked
from jaxloop import loop
from jaxloop import partition
# from jaxloop import pipeline_loop # Blocked
from jaxloop import stat_loop
from jaxloop import step
from jaxloop import train_loop
from jaxloop import types

# -- Project information

project = 'jaxloop'
copyright = '2024, The jaxloop Authors'
author = 'The jaxloop Authors'

release = '0.1'
version = '0.1.0'


print('sys.path:', sys.path)


# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
