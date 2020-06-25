# Installation script for python
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='eko',
      version='0.2.0',
      description='Evolution Kernel Operator',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author = 'S.Carrazza, J.Cruz-Martinez, F. Hekhorn',
      author_email='stefano.carrazza@cern.ch',
      url='https://github.com/N3PDF/eko',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      package_data = {
          '' : ['doc/source/img/Logo.svg'],
      },
      zip_safe=False,
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
          'cffi',
          'joblib',
          'pyyaml',
      ],
      setup_requires=[
          "cffi>1.0.0"
      ],
      cffi_modules=["src/cfunctions/digamma.py:ffibuilder"],
      python_requires='>=3.7'
)
