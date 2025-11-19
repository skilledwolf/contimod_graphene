import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'contimod_graphene'
copyright = '2024, Tobias Wolf'
author = 'Tobias Wolf'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
