"""Support legacy storage formats."""
import dataclasses
import io
import os
import pathlib
import tarfile
import tempfile
import warnings

import lz4.frame
import numpy as np
import numpy.typing as npt
import yaml

from eko.interpolation import XGrid
from eko.io.runcards import Rotations

from .. import basis_rotation as br
from . import raw
from .dictlike import DictLike
from .struct import EKO, Operator
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

    grid = op5to4(metaold["Q2grid"], arrays)

    with EKO.create(dest) as builder:
        # here I'm plainly ignoring the static analyzer, the types are faking
        # the actual ones - not sure if I should fix builder interface to
        # accept also these
        eko = builder.load_cards(theory, operator).build()  # pylint: disable=E1101

        for mu2, op in grid.items():
            eko[mu2] = op

        eko.metadata.version = metaold.get("eko_version", "")
        eko.metadata.data_version = 0
        eko.metadata.update()


@dataclasses.dataclass
class PseudoTheory(DictLike):
    """Fake theory, mocking :class:`eko.io.runcards.TheoryCard`.

    Used to provide a theory for the :class:`~eko.io.struct.EKO` builder, even when the theory
    information is not available.

    """

    _void: str

    @classmethod
    def from_old(cls, old: RawCard):
        """Load from old metadata."""
        return cls(_void="")


@dataclasses.dataclass
class PseudoOperator(DictLike):
    """Fake operator, mocking :class:`eko.io.runcards.OperatorCard`.

    Used to provide a theory for the :class:`~eko.io.struct.EKO` builder, even when the operator
    information is not fully available.

    """

    mu20: float
    mu2grid: npt.NDArray
    rotations: Rotations
    configs: dict

    @classmethod
    def from_old(cls, old: RawCard):
        """Load from old metadata."""
        mu20 = float(old["q2_ref"])
        mu2grid = np.array(old["Q2grid"])

        xgrid = XGrid(old["interpolation_xgrid"])
        pids = old.get("pids", np.array(br.flavor_basis_pids))

        rotations = Rotations(xgrid=xgrid, pids=pids)

        def set_if_different(name: str, default: npt.NDArray):
            basis = old.get(name)
            if basis is not None and not np.allclose(basis, default):
                setattr(rotations, name, basis)

        set_if_different("inputpids", pids)
        set_if_different("targetpids", pids)
        set_if_different("inputgrid", xgrid.raw)
        set_if_different("targetgrid", xgrid.raw)

        configs = dict(
            interpolation_polynomial_degree=old.get("interpolation_polynomial_degree"),
            interpolation_is_log=old.get("interpolation_is_log"),
        )

        return cls(mu20=mu20, mu2grid=mu2grid, rotations=rotations, configs=configs)


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


def op5to4(mu2grid: list, arrays: dict) -> dict:
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
    for mu2, op4, err4 in zip(mu2grid, op5, err5):
        grid[mu2] = Operator(operator=op4, error=err4)

    return grid
