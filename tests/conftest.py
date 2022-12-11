import dataclasses
import os
import pathlib
import shutil
import sys
from contextlib import contextmanager
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

from eko import interpolation
from eko.io.struct import EKO, AccessConfigs, Metadata, Operator, Rotations
from ekobox import cards


@pytest.fixture
def cd():
    # thanks https://stackoverflow.com/a/24176022/8653979
    @contextmanager
    def wrapped(newdir):
        prevdir = os.getcwd()
        os.chdir(os.path.expanduser(newdir))
        try:
            yield
        finally:
            os.chdir(prevdir)

    return wrapped


class FakePDF:
    def hasFlavor(self, pid):
        return pid == 1

    def xfxQ2(self, _pid, x, _q2):
        return x


@pytest.fixture
def fake_pdf():
    return FakePDF()


@pytest.fixture
def theory_card():
    return cards.example.theory()


@pytest.fixture()
def theory_ffns(theory_card):
    def set_(flavors: int):
        for q in "cbt":
            setattr(theory_card.matching, q, np.inf)
        return theory_card

    return set_


@pytest.fixture
def operator_card():
    card = cards.example.operator()
    card.rotations.xgrid = interpolation.XGrid([0.1, 0.3, 0.5, 1.0])
    card.configs.interpolation_polynomial_degree = 2

    return card


class EKOFactory:
    def __init__(self):
        self.cache: Optional[EKO] = None

    @staticmethod
    def _operators(mugrid: Sequence[float], shape: Tuple[int, int]):
        ops = {}
        for mu in mugrid:
            ops[2.0] = Operator(np.random.rand(*shape, *shape))

        return ops

    def _create(self, update: Optional[dict]):
        @dataclasses.dataclass
        class defaults:
            mu0 = 0.0
            mugrid = [2.0]
            xgrid = interpolation.XGrid([0.5, 1.0])
            pids = np.array([0, 1])
            path = pathlib.Path.cwd() / "test-eko-invalid-path"

        if update is not None:
            pars = defaults(**update)
        else:
            pars = defaults()

        lx = len(pars.xgrid)
        lpids = len(pars.pids)

        access = AccessConfigs(pars.path, False)
        metadata = Metadata(pars.mu0, Rotations(xgrid=pars.xgrid, pids=pars.pids))
        self.cache = EKO(
            _operators=self._operators(mugrid=pars.mugrid, shape=(lx, lpids)),
            access=access,
            metadata=metadata,
        )

        return self.cache

    def _clean(self):
        self.cache = None

    def get(self, update: Optional[dict] = None):
        """Get a fake output.

        To force a new output, keeping all the defaults, just pass an empty
        dictionary::

            factory.get({})


        Pay attention that the EKO created is an invalid one, since no path is
        actually connected.

        """
        if self.cache is None or update is not None:
            self.cache = self._create(update=update)

        return self.cache


@pytest.fixture
def eko_factory():
    factory = EKOFactory()
    yield factory
    factory._clean()


@pytest.fixture
def fake_lhapdf(tmp_path):
    def lhapdf_paths():
        return [tmp_path]

    # Thanks https://stackoverflow.com/questions/43162722/mocking-a-module-import-in-pytest
    module = type(sys)("lhapdf")
    module.paths = lhapdf_paths
    sys.modules["lhapdf"] = module

    yield tmp_path


fakepdf = pathlib.Path(__file__).parents[1] / "benchmarks" / "ekobox" / "fakepdf"


def copy_lhapdf(fake_lhapdf, n):
    src = fakepdf / n
    dst = fake_lhapdf / n
    shutil.copytree(src, dst)
    yield n
    shutil.rmtree(dst)


@pytest.fixture
def fake_ct14(fake_lhapdf):
    yield from copy_lhapdf(fake_lhapdf, "myCT14llo_NF3")


@pytest.fixture
def fake_nn31(fake_lhapdf):
    yield from copy_lhapdf(fake_lhapdf, "myNNPDF31_nlo_as_0118")


@pytest.fixture
def fake_mstw(fake_lhapdf):
    yield from copy_lhapdf(fake_lhapdf, "myMSTW2008nlo90cl")
