# -*- coding: utf-8 -*-
import pathlib

import packutil as pack
from setuptools import setup, find_packages

# write version on the fly - inspired by numpy
MAJOR = 0
MINOR = 6
MICRO = 0

repo_path = pathlib.Path(__file__).absolute().parent


def setup_package():
    # write version
    pack.versions.write_version_py(
        MAJOR,
        MINOR,
        MICRO,
        pack.versions.is_released(repo_path),
        filename="src/eko/version.py",
    )
    # paste Readme
    with open("README.md", "r") as fh:
        long_description = fh.read()
    # do it
    setup(
        name="eko",
        version=pack.versions.mkversion(MAJOR, MINOR, MICRO),
        description="Evolution Kernel Operator",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="A. Candido, S. Carrazza, J. Cruz-Martinez, F. Hekhorn, G. Magni",
        author_email="stefano.carrazza@cern.ch",
        url="https://github.com/N3PDF/eko",
        package_dir={"": "src"},
        packages=find_packages("src"),
        package_data={
            "": ["doc/source/img/Logo.png"],
        },
        classifiers=[
            "Operating System :: Unix",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Physics",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        ],
        install_requires=[
            "numpy",
            "scipy",
            "numba",
            "pyyaml",
            "lz4",
        ],
        setup_requires=["wheel"],
        python_requires=">=3.7",
    )


if __name__ == "__main__":
    setup_package()
