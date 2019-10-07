# Installation script for python
from setuptools import setup, find_packages

setup(name='eko',
      version='0.0.1',
      description='Evolution Kernel Operator',
      author = 'S.Carrazza, J.Cruz-Martinez, F. Hekhorn',
      author_email='stefano.carrazza@cern.ch',
      url='https://github.com/N3PDF/eko',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      zip_safe=False,
      classifiers=[
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      install_requires=[
          'numpy',
          'numba',
          'sphinx_rtd_theme',
      ],
      python_requires='>=3.6'
)
