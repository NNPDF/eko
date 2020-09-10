# -*- coding: utf-8 -*-
# Installation script for python
from setuptools import setup, find_packages

setup(
    name="ekomark",
    author="F. Hekhorn, A.Candido",
    version="0.1.0",
    description="eko benchmark",
    # package_dir={"": "."},
    packages=find_packages("."),
    install_requires=["matplotlib", "pandas"],
    entry_points={
        "console_scripts": [],
    },
    python_requires=">=3.7",
)
