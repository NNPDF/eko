# Installation script for python
import pathlib

from setuptools import setup

# find the shared object
so_name = pathlib.Path(__file__).parent.glob("msht_n3lo.*.so")
so_name = list(so_name)[0]

setup(
    name="msht_n3lo",
    author="",
    version="0.1.0",
    description="msht_n3lo",
    packages=[""],
    package_data={
        "": [str(so_name)],
    },
)
