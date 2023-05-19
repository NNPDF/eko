"""Support legacy storage formats."""
import dataclasses
import io
import os
import pathlib
import tarfile
import tempfile
import warnings
from typing import Dict, List

import lz4.frame
import numpy as np
import yaml

from eko.interpolation import XGrid
from eko.io.runcards import flavored_mugrid
from eko.quantities.heavy_quarks import HeavyInfo, HeavyQuarkMasses, MatchingRatios

from . import raw
from .dictlike import DictLike
from .struct import EKO, Operator
from .types import EvolutionPoint as EPoint
from .types import RawCard


def load_tar(source: os.PathLike, dest: os.PathLike, errors: bool = False):
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
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

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
        flavored_mugrid(op5, theory.heavy.masses, theory.heavy.matching_ratios), arrays
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
            num_flavs_init=4,
            num_flavs_max_pdf=None,
            intrinsic_flavors=None,
            masses=HeavyQuarkMasses([1.51, 4.92, 172.5]),
            masses_scheme=None,
            matching_ratios=MatchingRatios([1.0, 1.0, 1.0]),
        )
        return cls(heavy=heavy)


@dataclasses.dataclass
class PseudoOperator(DictLike):
    """Fake operator, mocking :class:`eko.io.runcards.OperatorCard`.

    Used to provide a theory for the :class:`~eko.io.struct.EKO` builder, even when the operator
    information is not fully available.

    """

    mu20: float
    evolgrid: List[EPoint]
    xgrid: XGrid
    configs: dict

    @classmethod
    def from_old(cls, old: RawCard):
        """Load from old metadata."""
        mu20 = float(old["q2_ref"])
        mu2list = old.get("Q2grid")
        if mu2list is None:
            mu2list = old["mu2grid"]
        mu2grid = np.array(mu2list)
        evolgrid = flavored_mugrid(
            np.sqrt(mu2grid).tolist(), [1.51, 4.92, 172.5], [1, 1, 1]
        )

        xgrid = XGrid(old["interpolation_xgrid"])

        configs = dict(
            interpolation_polynomial_degree=old.get("interpolation_polynomial_degree"),
            interpolation_is_log=old.get("interpolation_is_log"),
        )

        return cls(mu20=mu20, evolgrid=evolgrid, xgrid=xgrid, configs=configs)


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
