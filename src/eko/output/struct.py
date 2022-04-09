# -*- coding: utf-8 -*-
import contextlib
import io
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
import typing
from dataclasses import dataclass, fields
from typing import BinaryIO, Dict, Literal, Optional, Tuple

import lz4.frame
import numpy as np
import numpy.lib.npyio as npyio
import yaml

from .. import basis_rotation as br
from .. import interpolation
from .. import version as vmod

logger = logging.getLogger(__name__)

PathLike = typing.Union[str, os.PathLike]


class DictLike:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)

    @property
    def raw(self):
        dictionary = {}
        for field in fields(self):
            value = getattr(self, field.name)

            # replace numpy arrays with lists
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # replace numpy scalars with python ones
            elif isinstance(value, float):
                value = float(value)

            dictionary[field.name] = value

        return dictionary


@dataclass
class Operator(DictLike):
    alphas: float
    operator: np.ndarray
    error: Optional[np.ndarray] = None

    # IO works with streams in memory, in order to avoid intermediate write on
    # disk (keep read from and write to tar file only)

    def save(self, compress: bool = True) -> Tuple[BinaryIO, bool]:
        stream = io.BytesIO()
        if self.error is None:
            np.save(stream, self.operator)
        else:
            np.savez(stream, operator=self.operator, error=self.error)
        stream.seek(0)

        # compress if requested
        if compress:
            stream = io.BytesIO(lz4.frame.compress(stream.read()))

        # return the stream ready to be read, and the type of array dumped (i.e.
        # 'npy' or 'npz')
        return stream, self.error is None

    @classmethod
    def load(cls, stream: BinaryIO):
        content = np.load(stream)

        if isinstance(content, np.ndarray):
            op = content
            err = None
        elif isinstance(content, npyio.NpzFile):
            op = content["operator"]
            err = content["error"]
        else:
            raise ValueError(
                "Not possible to load operator, content format not recognized"
            )

        return cls(alphas=0.0, operator=op, error=err)


@dataclass
class Debug(DictLike):
    skip_singlet: bool = False
    skip_non_singlet: bool = False


@dataclass
class Configs(DictLike):
    ev_op_max_order: int
    ev_op_iterations: int
    interpolation_polynomial_degree: int
    interpolation_is_log: bool
    backward_inversion: Literal["exact", "expanded"]
    n_integration_cores: int = 1


@dataclass
class Rotations(DictLike):
    targetgrid: Optional[np.ndarray] = None
    inputgrid: Optional[np.ndarray] = None
    targetpids: Optional[np.ndarray] = None
    inputpids: Optional[np.ndarray] = None

    @classmethod
    def default(cls, xgrid: interpolation.XGrid, pids: np.ndarray):
        return cls(
            targetgrid=xgrid.raw, inputgrid=xgrid.raw, targetpids=pids, inputpids=pids
        )


@dataclass
class EKO:
    """Operator interface.

    This class offers an interface to an abstract operator, between memory and
    disk.

    An actual operator might be arbitrarily huge, and in particular size
    limitations in memory are far more strict than on disk.
    Since manually managing, for each application, the burden of off-loading
    part of the operator might be hard and occasionally not possible (without a
    clear picture of the internals), the library itself offers this facility.

    In particular, the data format on disk has a complete specification, and
    can hold a full operator independently of the loading procedure.
    In order to accomplish the former goal, the remaining task of partial
    loading is done by this class (for the Python library, other
    implementations are possible and encouraged).

    For this reason, a core component of an :cls:`EKO` object is a path,
    referring to the location on disk of the corresponding operator.
    Any :cls:`EKO` has an associated path:

    - for the computed object, it corresponds to the path where the actual
    result of the computation is already saved
    - for a new object, it is the path at which any result of final or
    intermediate computation is stored, as soon as it is produced

    The computation can be stopped at any time, without the loss of any of the
    intermediate results.

    Attributes
    ----------
    path : pathlib.Path
        path on disk, to which this object is linked (and for which it is
        essentially an interface)
    Q02 : float
        inital scale
    xgrid : interpolation.XGrid
        momentum fraction internal grid
    pids : np.ndarray
        array of integers, corresponding to internal PIDs
    configs : Configs
        specific configuration to be used during the calculation of these
        operators
    rotations : Rotations
        manipulation information, describing the current status of the EKO (e.g.
        `inputgrid` and `targetgrid`)
    debug : Debug
        debug configurations
    version : str
        library version used to create the corresponding file
    data_version : str
        specs version, to which the file adheres

    """

    # operators cache, contains the Q2 grid information
    _operators: Dict[float, Optional[Operator]]
    # public attributes
    # -----------------
    # mandatory, identifying features
    path: pathlib.Path
    Q02: float
    xgrid: interpolation.XGrid
    pids: np.ndarray
    # collections
    configs: Configs
    rotations: Rotations
    debug: Debug
    # tagging information
    version: str = vmod.__version__
    data_version: str = vmod.__data_version__

    def __post_init__(self):
        if self.path.suffix != ".tar":
            raise ValueError("Not a valid path for an EKO")
        if not tarfile.is_tarfile(self.path):
            raise ValueError("EKO: the corresponding file is not a valid tar archive")

    def __getitem__(self, q2: float):
        # TODO: autoload
        return self._operators[q2]

    def __setitem__(self, q2: float, op: Operator):
        # TODO: autodump
        if isinstance(op, dict):
            op = Operator.from_dict(op)
        if not isinstance(op, Operator):
            raise ValueError("Only operators can be stored.")

        self._operators[q2] = op

    def __delitem__(self, q2: float):
        """Drop operator from memory.

        This method only drops the operator from memory, and it's not expected
        to do anything else.

        Autosave is done on set, and explicit saves are performed by the
        computation functions.

        If a further explicit save is required, repeat explicit assignment::

            eko[q2] = eko[q2]

        This is only useful if the operator has been mutated in place, that in
        general should be avoided, since the operator should only be the result
        of a full computation or a library manipulation.


        Parameters
        ----------
        q2 : float
            the value of :math:`Q^2` for which the corresponding operator
            should be dropped

        """
        self._operators[q2] = None

    @contextlib.contextmanager
    def operator(self, q2: float):
        try:
            yield self[q2]
        finally:
            del self[q2]

    @property
    def Q2grid(self):
        return np.array(list(self._operators))

    def __iter__(self):
        """Iterate over keys (i.e. Q2 values)

        Yields
        ------
        float
            q2 values

        """
        for q2 in self._operators:
            yield q2

    def items(self):
        """Iterate operators, with minimal load.

        Pay attention, this iterator:

        - is not a read-only operation from the point of view of the in-memory
          object (since the final result after iteration is no operator loaded)
        - but it is a read-only operation from the point of view of the
          permanent object on-disk

        Yields
        ------
        tuple
            couples of ``(q2, operator)``, loaded immediately before, unloaded
            immediately after

        """
        for q2 in self.Q2grid:
            yield q2, self[q2]
            del self[q2]

    def __contains__(self, q2: float) -> bool:
        return q2 in self._operators

    def approx(self, q2, rtol=1e-6, atol=1e-10) -> Optional[float]:
        q2s = self.Q2grid
        close = q2s[np.isclose(q2, q2s, rtol=rtol, atol=atol)]

        if close.size == 1:
            return close[0]
        if close.size == 0:
            return None
        raise ValueError(f"Multiple values of Q2 have been found close to {q2}")

    @staticmethod
    def bootstrap(tdir: PathLike, theory: dict, operator: dict):
        tdir = pathlib.Path(tdir)
        (tdir / "theory.yaml").write_text(yaml.dump(theory), encoding="utf-8")
        (tdir / "operator.yaml").write_text(yaml.dump(operator), encoding="utf-8")
        (tdir / "recipes").mkdir()
        (tdir / "parts").mkdir()
        (tdir / "operators").mkdir()

    @staticmethod
    def extract(path: PathLike, filename: str) -> str:
        path = pathlib.Path(path)

        with tarfile.open(path, "r") as tar:
            fd = tar.extractfile(filename)
            if fd is None:
                raise ValueError(
                    f"The member '{filename}' is not a readable file inside EKO tar"
                )
            content = fd.read().decode()

        __import__("pdb").set_trace()
        return content

    @property
    def theory(self) -> dict:
        return yaml.safe_load(self.extract(self.path, "theory.yaml"))

    @property
    def theory_card(self) -> dict:
        return self.theory

    @property
    def operator_card(self) -> dict:
        return yaml.safe_load(self.extract(self.path, "operator.yaml"))

    @classmethod
    def detached(cls, theory: dict, operator: dict, path: pathlib.Path):
        """Build the in-memory object alone.

        Note
        ----
        This constructor is meant for internal use, backing the usual ones (like
        :meth:`new` or :meth:`load`), but it should not be used directly, since
        it has no guarantee that the underlying path is valid, breaking the
        object semantic.

        Parameters
        ----------
        theory : dict
            the theory card
        operator : dict
            the operator card
        path : str or os.PathLike
            the underlying path (it has to be a valid object, but it is not
            guaranteed, see the note)

        Returns
        -------
        EKO
            the generated structure

        """
        xgrid = interpolation.XGrid(operator["xgrid"])
        pids = np.array(operator.get("pids", np.array(br.flavor_basis_pids)))

        return cls(
            path=path,
            xgrid=xgrid,
            pids=pids,
            Q02=operator["Q0"] ** 2,
            _operators={q2: None for q2 in operator["Q2grid"]},
            configs=Configs.from_dict(operator["configs"]),
            rotations=Rotations.default(xgrid, pids),
            debug=Debug.from_dict(operator.get("debug", {})),
        )

    @classmethod
    def new(cls, theory: dict, operator: dict, path: Optional[PathLike] = None):
        """Make structure from runcard-like dictionary.

        This constructor is made to be used with loaded runcards, in order to
        minimize the amount of code needed to init a new object (you just to
        load the runcard and call this function).

        Note
        ----
        An object is initialized with no rotations, since the role of rotations
        is to keep the current state of the output object after manipulations
        happened.
        Since a new object is here in the process of being created, no rotation
        has to be logged.

        Parameters
        ----------
        theory : dict
            the theory card
        operator : dict
            the operator card
        path : str or os.PathLike
            the underlying path (if not provided, it is created in a temporary
            path)

        Returns
        -------
        EKO
            the generated structure

        """
        path = pathlib.Path(
            path if path is not None else tempfile.mkstemp(suffix=".tar")[1]
        )
        path.unlink()
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            cls.bootstrap(td, theory=theory, operator=operator)

            with tarfile.open(path, mode="w") as tar:
                for element in td.glob("*"):
                    tar.add(element, arcname=element.name)

            shutil.rmtree(td)

        eko = cls.detached(theory, operator, path=path)
        logger.info(f"New operator created at path '{path}'")
        return eko

    @classmethod
    def load(cls, path: PathLike):
        path = pathlib.Path(path)
        if not tarfile.is_tarfile(path):
            raise ValueError("EKO: the corresponding file is not a valid tar archive")

        theory = yaml.safe_load(cls.extract(path, "theory.yaml"))
        operator = yaml.safe_load(cls.extract(path, "operator.yaml"))

        eko = cls.detached(theory, operator, path=path)
        logger.info(f"Operator loaded from path '{path}'")
        return eko

    @property
    def raw(self):
        return dict(
            path=str(self.path),
            xgrid=self.xgrid.tolist(),
            pids=self.pids.tolist(),
            Q0=float(np.sqrt(self.Q02)),
            Q2grid=self.Q2grid,
            configs=self.configs.raw,
            rotations=self.rotations.raw,
            debug=self.debug.raw,
        )

    def close(self):
        for q2 in self.Q2grid:
            del self[q2]

    def __del__(self):
        self.close()
