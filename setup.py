# -*- coding: utf-8 -*-
# Installation script for python
from setuptools import setup, find_packages

# paste Readme
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='eko',
      version='0.3.0',
      description='Evolution Kernel Operator',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author = 'S. Carrazza, J. Cruz-Martinez, F. Hekhorn',
      author_email='stefano.carrazza@cern.ch',
      url='https://github.com/N3PDF/eko',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      package_data = {
          '' : ['doc/source/img/Logo.svg'],
      },
      classifiers=[
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      ],
      install_requires=[
          'numpy',
          'scipy',
          'numba',
          'joblib',
          'pyyaml',
      ],
      python_requires='>=3.7'
)
