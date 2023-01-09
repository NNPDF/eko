import os
import pathlib
import shutil
import sys
from contextlib import contextmanager
from typing import Iterable, Optional, Tuple

import numpy as np
import pytest

from eko import interpolation
from eko.io.runcards import OperatorCard, TheoryCard
from eko.io.struct import EKO, Operator
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
    def set_(flavors: int) -> TheoryCard:
        i = flavors - 3
        for q in "cbt"[i:]:
            setattr(theory_card.matching, q, np.inf)
        return theory_card

    return set_


@pytest.fixture
def operator_card():
    card = cards.example.operator()
    card.rotations.xgrid = interpolation.XGrid([0.1, 0.3, 0.5, 1.0])
    card.configs.interpolation_polynomial_degree = 2

    return card


@pytest.fixture
def out_v0():
    return pathlib.Path(__file__).parent / "data" / "v0.8.5-obf24af_t8e1305.tar"


class EKOFactory:
    def __init__(self, theory: TheoryCard, operator: OperatorCard, path: os.PathLike):
        self.path = path
        self.theory = theory
        self.operator = operator
        self.cache: Optional[EKO] = None

    @staticmethod
    def _operators(mugrid: Iterable[float], shape: Tuple[int, int]):
        ops = {}
        for mu in mugrid:
            ops[mu] = Operator(np.random.rand(*shape, *shape))

        return ops

    def _create(self):
        self.cache = (
            EKO.create(self.path).load_cards(self.theory, self.operator).build()
        )
        lx = len(self.operator.rotations.xgrid)
        lpids = len(self.operator.rotations.pids)
        for q2, op in self._operators(
            mugrid=self.operator.mu2grid, shape=(lpids, lx)
        ).items():
            self.cache[q2] = op

        return self.cache

    def _clean(self):
        if self.cache is not None:
            self.cache.close()
            self.cache.access.path.unlink()
        self.cache = None

    def get(self, update: Optional[dict] = None):
        """Get a fake output.

        To force a new output, keeping all the defaults, just pass an empty
        dictionary::

            factory.get({})

        """
        if self.cache is None or update is not None:
            self.cache = self._create()

        return self.cache


@pytest.fixture
def eko_factory(theory_ffns, operator_card, tmp_path: pathlib.Path):
    factory = EKOFactory(
        theory=theory_ffns(3), operator=operator_card, path=tmp_path / "eko.tar"
    )
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
