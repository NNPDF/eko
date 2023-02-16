import os
import tarfile

import pytest

from eko.io import raw


@pytest.fixture
def tar(tmp_path):
    tarpath = tmp_path / "eko.tar"

    myfile = tarfile.TarInfo(name="./ciao")
    with tarfile.open(tarpath, "w") as tar:
        tar.addfile(myfile)

    return tarpath


@pytest.fixture
def malicious_tar(tmp_path):
    tarpath = tmp_path / "eko.tar"

    myfile = tarfile.TarInfo(name="/ciao")
    with tarfile.open(tarpath, "w") as tar:
        tar.addfile(myfile)

    return tarpath


def test_nopath(tar, tmp_path):
    os.chdir(tmp_path)
    with tarfile.open(tar) as taro:
        raw.safe_extractall(taro)


def test_traversal(malicious_tar, tmp_path):
    with tarfile.open(malicious_tar) as taro:
        with pytest.raises(Exception, match="Path Traversal"):
            raw.safe_extractall(taro, path=tmp_path)
