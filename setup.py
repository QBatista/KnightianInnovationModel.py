# Use setuptools in preference to distutils
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import os

DESCRIPTION = "A package for solving the Knightian Model of Innovation"

setup(name='knightian_model',
      packages=['knightian_model'],
      version=0.1,
      description=DESCRIPTION,
      author='Quentin Batista',
      author_email='batista.quent@gmail.com',
      url='https://github.com/QBatista/knightian_model',  # URL to the repo
      keywords=['quantitative', 'economics', 'knightian'],
      install_requires=[
          'numba>=0.38',
          'numpy',
          'quantecon',
          'interpolation',
          ]
      )
