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
    install_requires=["numpy","matplotlib", "pandas","banana-hep"],
    entry_points={
        "console_scripts": [
            "generate_theories=ekomark.data:generate_theories",
            "generate_operators=ekomark.data:generate_operators",
        ],
    },
    python_requires=">=3.7",
)
