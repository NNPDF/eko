# -*- coding: utf-8 -*-
"""
Test the generation of version.py by setup.py.
"""
import pathlib

import pygit2
import semver

import eko.version as v

repo_path = pathlib.Path(__file__).absolute().parents[1]
repo = pygit2.Repository(repo_path)


class TestVersion:
    def test_version(self):
        tags = [ref.split("/")[-1] for ref in repo.references if "/tags/" in ref]

        versions = []
        for tag in tags:
            try:
                versions.append(semver.VersionInfo.parse(tag[1:]))
            except ValueError:
                # if tag is not following semver do not append
                pass

        last_version = max(versions)
        assert v.major == last_version.major
        assert v.short_version == f"{last_version.major}.{last_version.minor}"
        assert v.version == str(last_version)
        assert v.version == v.full_version.split("-")[0]

    def test_released(self):
        release_branches = ["master", "release", "hotfix"]
        branch_name = "/".join(repo.head.name.split("/")[2:])

        full_version = semver.VersionInfo.parse(v.full_version)

        if branch_name.split("/")[0] in release_branches:
            assert v.is_released
            assert full_version.prerelease is None
        else:
            assert not v.is_released
            assert full_version.prerelease == "develop"
