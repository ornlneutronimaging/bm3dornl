import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'bm3dornl'
copyright = '2025, ORNL Neutron Imaging'
author = 'Chen Zhang'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

html_theme = 'sphinx_rtd_theme'

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Mock imports for ReadTheDocs (Rust extension won't be available)
autodoc_mock_imports = ['bm3dornl.bm3d_rust']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
