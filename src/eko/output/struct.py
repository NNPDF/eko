"""Define output representation structures."""
import contextlib
import copy
import io
import logging
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, fields
from typing import BinaryIO, Dict, Literal, Optional, Tuple

import lz4.frame
import numpy as np
import numpy.lib.npyio as npyio
import numpy.typing as npt
import yaml

from .. import basis_rotation as br
from .. import interpolation
from .. import version as vmod

logger = logging.getLogger(__name__)

THEORYFILE = "theory.yaml"
OPERATORFILE = "operator.yaml"
RECIPESDIR = "recipes"
PARTSDIR = "parts"
OPERATORSDIR = "operators"


class DictLike:
    """Dictionary compatibility base class, for dataclasses.

    This class add compatibility to import and export from Python :class:`dict`,
    in such a way to support serialization interfaces working with them.

    Some collections and scalar objects are normalized to native Python
    structures, in order to simplify the on-disk representation.

    """

    def __init__(self, **kwargs):
        """Empty initializer."""

    @classmethod
    def from_dict(cls, dictionary):
        """Initialize dataclass object from raw dictionary.

        Parameters
        ----------
        dictionary: dict
            the dictionary to be converted to :class:`DictLike`

        Returns
        -------
        DictLike
            instance with `dictionary` content loaded as attributes

        """
        return cls(**dictionary)

    @property
    def raw(self):
        """Convert dataclass object to raw dictionary.

        Normalize:

            - :class:`np.ndarray` to lists (possibly nested)
            - scalars to the corresponding built-in type (e.g. :class:`float`)
            - :class:`tuple` to lists
            - :class:`interpolation.XGrid` to the intrinsic serialization format

        Returns
        -------
        dict
            dictionary representation

        """
        dictionary = {}
        for field in fields(self):
            value = getattr(self, field.name)

            # replace numpy arrays with lists
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # replace numpy scalars with python ones
            elif isinstance(value, float):
                value = float(value)
            elif isinstance(value, interpolation.XGrid):
                value = value.dump()
            elif isinstance(value, tuple):
                value = list(value)

            dictionary[field.name] = value

        return dictionary


@dataclass
class Operator(DictLike):
    """Operator representation.

    To be used to hold the result of a computed 4-dim operator (from a given
    scale to another given one).

    Note
    ----
    IO works with streams in memory, in order to avoid intermediate write on
    disk (keep read from and write to tar file only).

    """

    operator: npt.NDArray
    """Content of the evolution operator."""
    error: Optional[npt.NDArray] = None
    """Errors on individual operator elements (mainly used for integration
    error, but it can host any kind of error).
    """

    def save(self, stream: BinaryIO, compress: bool = True) -> bool:
        """Save content of operator to bytes.

        Parameters
        ----------
        stream: BinaryIO
            a stream where to save the operator content (in order to be able to
            perform the operation both on disk and in memory)
        compress: bool
            flag to control optional compression (default: `True`)

        Returns
        -------
        bool
            whether the operator saved contained or not the error (this control
            even the format, ``npz`` with errors, ``npy`` otherwise)

        """
        if self.error is None:
            np.save(stream, self.operator)
        else:
            np.savez(stream, operator=self.operator, error=self.error)
        stream.seek(0)

        # compress if requested
        if compress:
            compressed = lz4.frame.compress(stream.read())
            # remove previous content
            stream.seek(0)
            stream.truncate()
            # write compressed data
            stream.write(compressed)
            stream.seek(0)

        # return the type of array dumped (i.e. 'npy' or 'npz')
        return self.error is None

    @classmethod
    def load(cls, stream: BinaryIO, compressed: bool = True):
        """Load operator from bytes.

        Parameters
        ----------
        stream: BinaryIO
            a stream to load the operator from (to support the operation both on
            disk and in memory)
        compressed: bool
            declare whether the stream is or is not compressed (default: `True`)

        Returns
        -------
        Operator
            the loaded instance

        """
        if compressed:
            stream = io.BytesIO(lz4.frame.decompress(stream.read()))
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

        return cls(operator=op, error=err)


@dataclass
class Debug(DictLike):
    """Debug configurations."""

    skip_singlet: bool = False
    """Whether to skip QCD singlet computation."""
    skip_non_singlet: bool = False
    """Whether to skip QCD non-singlet computation."""


@dataclass
class Configs(DictLike):
    """Solution specific configurations."""

    ev_op_max_order: Tuple[int]
    """Maximum order to use in U matrices expansion.
    Used only in ``perturbative`` solutions.
    """
    ev_op_iterations: int
    """Number of intervals in which to break the global path."""
    interpolation_polynomial_degree: int
    """Degree of elements of the intepolation polynomial basis."""
    interpolation_is_log: bool
    r"""Whether to use polynomials in :math:`\log(x)`.
    If `false`, polynomials are in :math:`x`.
    """
    backward_inversion: Literal["exact", "expanded"]
    """Which method to use for backward matching conditions."""
    n_integration_cores: int = 1
    """Number of cores used to parallelize integration."""


@dataclass
class Rotations(DictLike):
    """Rotations related configurations.

    Here "Rotation" is intended in a broad sense: it includes both rotations in
    flavor space (labeled with suffix `pids`) and in :math:`x`-space (labeled
    with suffix `grid`).
    Rotations in :math:`x`-space correspond to reinterpolate the result on a
    different basis of polynomials.

    """

    xgrid: interpolation.XGrid
    """Momentum fraction internal grid."""
    pids: npt.NDArray
    """Array of integers, corresponding to internal PIDs."""
    _targetgrid: Optional[interpolation.XGrid] = None
    _inputgrid: Optional[interpolation.XGrid] = None
    _targetpids: Optional[npt.NDArray] = None
    _inputpids: Optional[npt.NDArray] = None

    def __post_init__(self):
        """Adjust types when loaded from serialized object."""
        for attr in ("xgrid", "_inputgrid", "_targetgrid"):
            value = getattr(self, attr)
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                setattr(self, attr, interpolation.XGrid(value))
            elif not isinstance(value, interpolation.XGrid):
                setattr(self, attr, interpolation.XGrid.load(value))

    @property
    def inputpids(self) -> npt.NDArray:
        """Provide pids expected on the input PDF."""
        if self._inputpids is None:
            return self.pids
        return self._inputpids

    @property
    def targetpids(self) -> npt.NDArray:
        """Provide pids corresponding to the output PDF."""
        if self._targetpids is None:
            return self.pids
        return self._targetpids

    @property
    def inputgrid(self) -> interpolation.XGrid:
        """Provide :math:`x`-grid expected on the input PDF."""
        if self._inputgrid is None:
            return self.xgrid
        return self._inputgrid

    @property
    def targetgrid(self) -> interpolation.XGrid:
        """Provide :math:`x`-grid corresponding to the output PDF."""
        if self._targetgrid is None:
            return self.xgrid
        return self._targetgrid


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

    For this reason, a core component of an :class:`EKO` object is a path,
    referring to the location on disk of the corresponding operator.
    Any :class:`EKO` has an associated path:

        - for the computed object, it corresponds to the path where the actual
          result of the computation is already saved
        - for a new object, it is the path at which any result of final or
          intermediate computation is stored, as soon as it is produced

    The computation can be stopped at any time, without the loss of any of the
    intermediate results.

    """

    # operators cache, contains the Q2 grid information
    _operators: Dict[float, Optional[Operator]]
    # public attributes
    # -----------------
    # mandatory, identifying features
    path: pathlib.Path
    """Path on disk, to which this object is linked (and for which it is
    essentially an interface).
    """
    Q02: float
    """Inital scale."""
    # collections
    configs: Configs
    """Specific configuration to be used during the calculation of these
    operators.
    """
    rotations: Rotations
    """Manipulation information, describing the current status of the EKO (e.g.
    `inputgrid` and `targetgrid`).
    """
    debug: Debug
    """Debug configurations."""
    # tagging information
    version: str = vmod.__version__
    """Library version used to create the corresponding file."""
    data_version: str = vmod.__data_version__
    """Specs version, to which the file adheres."""

    @property
    def xgrid(self) -> interpolation.XGrid:
        """Momentum fraction internal grid."""
        return self.rotations.xgrid

    @xgrid.setter
    def xgrid(self, value: interpolation.XGrid):
        """Set `xgrid` value."""
        self.rotations.xgrid = value

    def __post_init__(self):
        """Validate class members."""
        if self.path.suffix != ".tar":
            raise ValueError("Not a valid path for an EKO")

    @staticmethod
    def opname(q2: float) -> str:
        """Operator file name from :math:`Q^2` value."""
        return f"{OPERATORSDIR}/{q2:8.2f}"

    def __getitem__(self, q2: float) -> Operator:
        """Retrieve operator for given :math:`Q^2`.

        If the operator is not already in memory, it will be automatically
        loaded.

        Parameters
        ----------
        q2 : float
            :math:`Q^2` value labeling the operator to be retrieved

        Returns
        -------
        Operator
            the retrieved operator

        """
        if q2 in self._operators:
            op = self._operators[q2]
            if op is not None:
                return op

        with tarfile.open(self.path) as tar:
            names = list(
                filter(lambda n: n.startswith(self.opname(q2)), tar.getnames())
            )

            if len(names) == 0:
                raise ValueError(f"Q2 value '{q2}' not available in '{self.path}'")

            name = names[0]
            compressed = name.endswith(".lz4")
            stream = tar.extractfile(name)

            op = Operator.load(stream, compressed=compressed)

        self._operators[q2] = op
        return op

    def __setitem__(self, q2: float, op: Operator, compress: bool = True):
        """Set operator for given :math:`Q^2`.

        The operator is automatically dumped on disk, .

        Parameters
        ----------
        q2: float
            :math:`Q^2` value labeling the operator to be set
        op: Operator
            the retrieved operator
        compress: bool
            whether to save the operator compressed or not (default: `True`)

        """
        if not isinstance(op, Operator):
            raise ValueError("Only an Operator can be added to an EKO")

        stream = io.BytesIO()
        without_err = op.save(stream, compress)
        stream.seek(0)

        suffix = "npy" if without_err else "npz"
        if compress:
            suffix += ".lz4"

        info = tarfile.TarInfo(name=f"{self.opname(q2)}.{suffix}")
        info.size = len(stream.getbuffer())
        info.mtime = int(time.time())
        info.mode = 436
        #  info.uname = os.getlogin()
        #  info.gname = os.getlogin()

        # TODO: unfortunately Python has no native support for deleting
        # files inside tar, so the proper way is to make that function
        # ourselves, in the inefficient way of constructing a new archive
        # from the existing one, but for the file to be removed
        # at the moment, an implicit dependency on `tar` command has been
        # introduced -> dangerous for portability
        # since it's not raising any error, it is fine to run in any case:
        has_file = False
        with tarfile.open(self.path, mode="r") as tar:
            has_file = f"operators/{q2:8.2f}.{suffix}" in tar.getnames()

        if has_file:
            subprocess.run(
                f"tar -f {self.path.absolute()} --delete".split() + [info.name]
            )
        with tarfile.open(self.path, "a") as tar:
            tar.addfile(info, fileobj=stream)

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
        """Retrieve operator and discard immediately.

        To be used as a contextmanager: the operator is automatically loaded as
        usual, but after the context manager it is dropped from memory.

        Parameters
        ----------
        q2: float
            :math:`Q^2` value labeling the operator to be retrieved

        """
        try:
            yield self[q2]
        finally:
            del self[q2]

    @property
    def Q2grid(self) -> npt.NDArray:
        """Provide the list of :math:`Q^2` as an array."""
        return np.array(list(self._operators))

    def __iter__(self):
        """Iterate over keys (i.e. Q2 values).

        Yields
        ------
        float
            q2 values

        """
        yield from self._operators

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
        """Check whether :math:`Q^2` operators are present.

        'Present' means, in this case, they are conceptually part of the
        :class:`EKO`. But it is telling nothing about being loaded in memory or
        not.

        Returns
        -------
        bool
            the result of checked condition

        """
        return q2 in self._operators

    def approx(
        self, q2: float, rtol: float = 1e-6, atol: float = 1e-10
    ) -> Optional[float]:
        """Look for close enough :math:`Q^2` value in the :class:`EKO`.

        Parameters
        ----------
        q2: float
            value of :math:`Q2` in which neighborhood to look
        rtol: float
            relative tolerance
        atol: float
            absolute tolerance

        Returns
        -------
        float or None
            retrieved value of :math:`Q^2`, if a single one is found

        Raises
        ------
        ValueError
            if multiple values are find in the neighborhood

        """
        q2s = self.Q2grid
        close = q2s[np.isclose(q2, q2s, rtol=rtol, atol=atol)]

        if close.size == 1:
            return close[0]
        if close.size == 0:
            return None
        raise ValueError(f"Multiple values of Q2 have been found close to {q2}")

    def unload(self):
        """Fully unload the operators in memory."""
        for q2 in self:
            del self[q2]

    def deepcopy(self, path: os.PathLike):
        """Create a deep copy of current instance.

        The managed on-disk object is copied as well, to the new ``path``
        location.
        If you don't want to copy the disk, consider using directly::

            copy.deepcopy(myeko)

        It will perform the exact same operation, without propagating it to the
        disk counterpart.

        Parameters
        ----------
        path: os.PathLike
            path were to copy the disk counterpart of the operator object

        Returns
        -------
        EKO
            the copy created

        """
        # return the object fully on disk, in order to avoid expensive copies in
        # memory (they would be copied on-disk in any case)
        self.unload()

        new = copy.deepcopy(self)
        new.path = pathlib.Path(path)
        shutil.copy2(self.path, new.path)

        return new

    @staticmethod
    def bootstrap(dirpath: os.PathLike, theory: dict, operator: dict):
        """Create directory structure.

        Parameters
        ----------
        dirpath: os.PathLike
            path to create the directory into
        theory: dict
            theory card to be dumped
        operator: dict
            operator card to be dumped

        """
        dirpath = pathlib.Path(dirpath)
        (dirpath / THEORYFILE).write_text(yaml.dump(theory), encoding="utf-8")
        (dirpath / OPERATORFILE).write_text(yaml.dump(operator), encoding="utf-8")
        (dirpath / RECIPESDIR).mkdir()
        (dirpath / PARTSDIR).mkdir()
        (dirpath / OPERATORSDIR).mkdir()

    @staticmethod
    def extract(path: os.PathLike, filename: str) -> str:
        """Extract file from disk assets.

        Note
        ----
        At the moment, it only support text files (since it is returning the
        content as a string)

        Parameters
        ----------
        path: os.PathLike
            path to the disk dump
        filename: str
            relative path inside the archive of the file to extract

        Returns
        -------
        str
            file content

        """
        path = pathlib.Path(path)

        with tarfile.open(path, "r") as tar:
            fd = tar.extractfile(filename)
            content = fd.read().decode()

        return content

    @property
    def theory(self) -> dict:
        """Provide theory card, retrieving from the dump."""
        # TODO: make an actual attribute, move load to `theory_card`, and the
        # type of the attribute will be `eko.runcards.TheoryCard`
        return yaml.safe_load(self.extract(self.path, THEORYFILE))

    @property
    def theory_card(self) -> dict:
        """Provide theory card, retrieving from the dump."""
        # TODO: return `eko.runcards.TheoryCard`
        return self.theory

    @property
    def operator_card(self) -> dict:
        """Provide operator card, retrieving from the dump."""
        # TODO: return `eko.runcards.OperatorCard`
        return yaml.safe_load(self.extract(self.path, OPERATORFILE))

    def interpolator(
        self, mode_N: bool, use_target: bool
    ) -> interpolation.InterpolatorDispatcher:
        """Return associated interpolation.

        Paramters
        ---------
        mode_N : bool
            interpolate in N-space?
        use_target : bool
            use target grid? If False, use input grid

        Returns
        -------
        interpolation.InterpolatorDispatcher
            interpolator
        """
        grid = self.rotations.targetgrid if use_target else self.rotations.inputgrid
        return interpolation.InterpolatorDispatcher(
            grid, self.configs.interpolation_polynomial_degree, mode_N
        )

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
        path : os.PathLike
            the underlying path (it has to be a valid object, but it is not
            guaranteed, see the note)

        Returns
        -------
        EKO
            the generated structure

        """
        bases = operator["rotations"]
        bases["pids"] = np.array(br.flavor_basis_pids)
        if operator["rotations"]["xgrid"] is not None:
        	bases["xgrid"] = interpolation.XGrid(
                	operator["rotations"]["xgrid"],
                	log=operator["configs"]["interpolation_is_log"],
            	)
        return cls(
            path=path,
            Q02=float(operator["Q0"] ** 2),
            _operators={q2: None for q2 in operator["Q2grid"]},
            configs=Configs.from_dict(operator["configs"]),
            rotations=Rotations.from_dict(bases),
            debug=Debug.from_dict(operator.get("debug", {})),
        )

    @classmethod
    def new(cls, theory: dict, operator: dict, path: Optional[os.PathLike] = None):
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
        path : os.PathLike
            the underlying path (if not provided, it is created in a temporary
            path)

        Returns
        -------
        EKO
            the generated structure

        """
        givenpath = path is not None

        path = pathlib.Path(path if givenpath else tempfile.mkstemp(suffix=".tar")[1])
        if path.exists():
            if givenpath:
                raise FileExistsError(
                    f"File exists at given path '{path}', cannot be used for a new operator."
                )
            # delete the file created in case of temporary file
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
    def load(cls, path: os.PathLike):
        """Load dump into an :class:`EKO` object.

        Parameters
        ----------
        path:: os.PathLike
            path to the dump to load

        Returns
        -------
        EKO
            the loaded instance

        """
        path = pathlib.Path(path)
        if not tarfile.is_tarfile(path):
            raise ValueError("EKO: the corresponding file is not a valid tar archive")

        theory = yaml.safe_load(cls.extract(path, THEORYFILE))
        operator = yaml.safe_load(cls.extract(path, OPERATORFILE))

        eko = cls.detached(theory, operator, path=path)
        logger.info(f"Operator loaded from path '{path}'")
        return eko

    @property
    def raw(self) -> dict:
        """Provide raw representation of the full content.

        Returns
        -------
        dict
            nested dictionary, storing all the values in the structure, but the
            operators themselves

        """
        return dict(
            path=str(self.path),
            Q0=float(np.sqrt(self.Q02)),
            Q2grid=self.Q2grid.tolist(),
            configs=self.configs.raw,
            rotations=self.rotations.raw,
            debug=self.debug.raw,
        )

    def __del__(self):
        """Destroy the memory structure gracefully."""
        self.unload()
