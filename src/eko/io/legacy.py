"""Support legacy storage formats."""

import dataclasses
import io
import pathlib
import tarfile
import tempfile
import warnings
from typing import Dict, List

import lz4.frame
import numpy as np
import yaml

from ..interpolation import XGrid
from ..io.runcards import flavored_mugrid
from ..quantities.heavy_quarks import (
    HeavyInfo,
    HeavyQuarkMasses,
    MatchingRatios,
    QuarkMassScheme,
)
from . import raw
from .dictlike import DictLike
from .struct import EKO, Operator
from .types import EvolutionPoint as EPoint
from .types import RawCard, ReferenceRunning

_MC = 1.51
_MB = 4.92
_MT = 172.5


def load_tar(source: pathlib.Path, dest: pathlib.Path, errors: bool = False):
    """Load tar representation from file.

    Compliant with :meth:`dump_tar` output.

    Parameters
    ----------
    source :
        source tar name
    dest :
        dest tar name
    errors :
        whether to load also errors (default ``False``)
    """
    with tempfile.TemporaryDirectory() as tmpdirr:
        tmpdir = pathlib.Path(tmpdirr)

        with tarfile.open(source, "r") as tar:
            raw.safe_extractall(tar, tmpdir)

        # load metadata
        innerdir = list(tmpdir.glob("*"))[0]
        metapath = innerdir / "metadata.yaml"
        metaold = yaml.safe_load(metapath.read_text(encoding="utf-8"))

        theory = PseudoTheory.from_old(metaold)
        operator = PseudoOperator.from_old(metaold)

        # get actual grids
        arrays = load_arrays(innerdir)

    op5 = metaold.get("Q2grid")
    if op5 is None:
        op5 = metaold["mu2grid"]
    grid = op5to4(
        flavored_mugrid(op5, [_MC, _MB, _MT], theory.heavy.matching_ratios), arrays
    )

    with EKO.create(dest) as builder:
        # here I'm plainly ignoring the static analyzer, the types are faking
        # the actual ones - not sure if I should fix builder interface to
        # accept also these
        eko = builder.load_cards(theory, operator).build()  # pylint: disable=E1101

        for ep, op in grid.items():
            eko[ep] = op

        eko.metadata.version = metaold.get("eko_version", "")
        eko.metadata.data_version = 0
        eko.metadata.update()


@dataclasses.dataclass
class PseudoTheory(DictLike):
    """Fake theory, mocking :class:`eko.io.runcards.TheoryCard`.

    Used to provide a theory for the :class:`~eko.io.struct.EKO` builder, even when the theory
    information is not available.
    """

    heavy: HeavyInfo

    @classmethod
    def from_old(cls, old: RawCard):
        """Load from old metadata."""
        heavy = HeavyInfo(
            masses=HeavyQuarkMasses(
                [
                    ReferenceRunning([_MC, np.inf]),
                    ReferenceRunning([_MB, np.inf]),
                    ReferenceRunning([_MT, np.inf]),
                ]
            ),
            masses_scheme=QuarkMassScheme.POLE,
            matching_ratios=MatchingRatios([1.0, 1.0, 1.0]),
        )
        return cls(heavy=heavy)


@dataclasses.dataclass
class PseudoOperator(DictLike):
    """Fake operator, mocking :class:`eko.io.runcards.OperatorCard`.

    Used to provide a theory for the :class:`~eko.io.struct.EKO` builder, even when the operator
    information is not fully available.
    """

    init: EPoint
    evolgrid: List[EPoint]
    xgrid: XGrid
    configs: dict

    @classmethod
    def from_old(cls, old: RawCard):
        """Load from old metadata."""
        mu0 = float(np.sqrt(float(old["q2_ref"])))
        mu2list = old.get("Q2grid")
        if mu2list is None:
            mu2list = old["mu2grid"]
        mu2grid = np.array(mu2list)
        evolgrid = flavored_mugrid(
            np.sqrt(mu2grid).tolist(), [_MC, _MB, _MT], [1, 1, 1]
        )

        xgrid = XGrid(old["interpolation_xgrid"])

        configs = dict(
            interpolation_polynomial_degree=old.get("interpolation_polynomial_degree"),
            interpolation_is_log=old.get("interpolation_is_log"),
        )

        return cls(init=(mu0, 4), evolgrid=evolgrid, xgrid=xgrid, configs=configs)


ARRAY_SUFFIX = ".npy.lz4"
"""Suffix for array files inside the tarball."""


def load_arrays(dir: pathlib.Path) -> dict:
    """Load arrays from compressed dumps."""
    arrays = {}
    for fp in dir.glob(f"*{ARRAY_SUFFIX}"):
        with lz4.frame.open(fp, "rb") as fd:
            # static analyzer can not guarantee the content to be bytes
            content = fd.read()
            assert not isinstance(content, str), "Bytes expected"

            stream = io.BytesIO(content)
            stream.seek(0)
            arrays[pathlib.Path(fp.stem).stem] = np.load(stream)

    return arrays


OPERATOR = "operator"
"""File name stem for operators."""
ERROR = "operator_error"
"""File name stem for operator errrors."""


def op5to4(evolgrid: List[EPoint], arrays: dict) -> Dict[EPoint, Operator]:
    """Load dictionary of 4-dim operators, from a single 5-dim one."""

    def plural(name: str) -> list:
        return [name, f"{name}s"]

    op5 = None  # 5 dimensional operator
    err5 = None
    for name, ar in arrays.items():
        if name in plural(OPERATOR):
            op5 = ar
        elif name in plural(ERROR):
            err5 = ar
        else:
            warnings.warn(f"Unrecognized array loaded, '{name}'")

    if op5 is None:
        raise RuntimeError("Operator not found")
    if err5 is None:
        err5 = [None] * len(op5)

    grid = {}
    for ep, op4, err4 in zip(evolgrid, op5, err5):
        grid[ep] = Operator(operator=op4, error=err4)

    return grid
