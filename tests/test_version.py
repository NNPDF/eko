# -*- coding: utf-8 -*-
"""
Test the generation of version.py by setup.py.
"""
import pathlib

import packutil.versions

import eko.version as v

repo_path = pathlib.Path(__file__).absolute().parents[1]


class TestVersion:
    def test_version(self):
        packutil.versions.test_version(repo_path, v)

    def test_released(self):
        packutil.versions.test_released(repo_path, v)
