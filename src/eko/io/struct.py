"""Define output representation structures."""
import contextlib
import copy
import io
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from typing import BinaryIO, Dict, Optional

import lz4.frame
import numpy as np
import numpy.lib.npyio as npyio
import numpy.typing as npt
import yaml

from .. import basis_rotation as br
from .. import interpolation
from .. import version as vmod
from . import exceptions
from .dictlike import DictLike
from .runcards import Configs, Debug, OperatorCard, Rotations

logger = logging.getLogger(__name__)

THEORYFILE = "theory.yaml"
OPERATORFILE = "operator.yaml"
METADATAFILE = "metadata.yaml"
RECIPESDIR = "recipes"
PARTSDIR = "parts"
OPERATORSDIR = "operators"

COMPRESSED_SUFFIX = ".lz4"


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
        stream : BinaryIO
            a stream where to save the operator content (in order to be able to
            perform the operation both on disk and in memory)
        compress : bool
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
        stream : BinaryIO
            a stream to load the operator from (to support the operation both on
            disk and in memory)
        compressed : bool
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
            raise exceptions.OperatorLoadingError(
                "Not possible to load operator, content format not recognized"
            )

        return cls(operator=op, error=err)


@dataclass
class InternalPaths:
    """Paths inside an EKO folder.

    This structure exists to locate in a single place the internal structure of
    an EKO folder.

    The only value required is the root path, everything else is computed
    relative to this root.
    In case only the relative paths are required, just create this structure
    with :attribute:`root` equal to emtpty string or ``"."``.

    """

    root: pathlib.Path
    "The root of the EKO folder (use placeholder if not relevant)"

    @property
    def metadata(self):
        """Metadata file."""
        return self.root / METADATAFILE

    @property
    def recipes(self):
        """Recipes folder."""
        return self.root / RECIPESDIR

    @property
    def parts(self):
        """Parts folder."""
        return self.root / PARTSDIR

    @property
    def operators(self):
        """Operators folder.

        This is the one containing the actual EKO components, after
        computation has been performed.

        """
        return self.root / OPERATORSDIR

    @property
    def theory_card(self):
        """Theory card dump."""
        return self.root / THEORYFILE

    @property
    def operator_card(self):
        """Operator card dump."""
        return self.root / OPERATORFILE


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
    _path: pathlib.Path
    """Path on disk, to which this object is linked (and for which it is
    essentially an interface).
    """
    Q02: float
    """Inital scale."""
    rotations: Rotations
    """Manipulation information, describing the current status of the EKO (e.g.
    `inputgrid` and `targetgrid`).
    """
    # tagging information
    version: str = vmod.__version__
    """Library version used to create the corresponding file."""
    data_version: str = vmod.__data_version__
    """Specs version, to which the file adheres."""

    @property
    def paths(self) -> InternalPaths:
        """Accessor for internal paths."""
        return InternalPaths(self._path)

    @property
    def xgrid(self) -> interpolation.XGrid:
        """Momentum fraction internal grid."""
        return self.rotations.xgrid

    @xgrid.setter
    def xgrid(self, value: interpolation.XGrid):
        """Set `xgrid` value."""
        self.rotations.xgrid = value

    @staticmethod
    def opname(q2: float) -> str:
        """Operator file name from :math:`Q^2` value."""
        return f"{q2:8.2f}"

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

        name = self.opname(q2)
        op_paths = list(
            filter(
                lambda path: path.name.startswith(name), self.paths.operators.iterdir()
            )
        )

        if len(op_paths) == 0:
            raise ValueError(f"Q2 value '{q2}' not available.")
        elif len(op_paths) > 1:
            raise ValueError(f"Too many operators associated to '{q2}'")

        op_path = op_paths[0]
        compressed = op_path.suffix == COMPRESSED_SUFFIX

        with open(op_path, "rb") as fd:
            op = Operator.load(fd, compressed=compressed)

        self._operators[q2] = op
        return op

    def __setitem__(self, q2: float, op: Operator, compress: bool = True):
        """Set operator for given :math:`Q^2`.

        The operator is automatically dumped on disk, .

        Parameters
        ----------
        q2 : float
            :math:`Q^2` value labeling the operator to be set
        op : Operator
            the retrieved operator
        compress : bool
            whether to save the operator compressed or not (default: `True`)

        """
        if not isinstance(op, Operator):
            raise ValueError("Only an Operator can be added to an EKO")

        without_err = op.error is None
        suffix = "npy" if without_err else "npz"
        if compress:
            suffix += COMPRESSED_SUFFIX

        op_path = (self.paths.operators / self.opname(q2)).with_suffix(suffix)

        with open(op_path, "wb") as fd:
            without_err2 = op.save(fd, compress)
        assert without_err == without_err2

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
        """Retrieve an operator and discard it afterwards.

        To be used as a contextmanager: the operator is automatically loaded as
        usual, but on the closing of the context manager it is dropped from
        memory.

        Parameters
        ----------
        q2 : float
            :math:`Q^2` value labeling the operator to be retrieved

        """
        try:
            yield self[q2]
        finally:
            del self[q2]

    @property
    def mu2grid(self) -> npt.NDArray:
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
        for q2 in self.mu2grid:
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
        q2 : float
            value of :math:`Q2` in which neighbourhood to look
        rtol : float
            relative tolerance
        atol : float
            absolute tolerance

        Returns
        -------
        float or None
            retrieved value of :math:`Q^2`, if a single one is found

        Raises
        ------
        ValueError
            if multiple values are find in the neighbourhood

        """
        q2s = self.mu2grid
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
        path : os.PathLike
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
    def bootstrap(dirpath: os.PathLike, theory: dict, operator: dict, metadata: dict):
        """Create directory structure.

        Parameters
        ----------
        dirpath : os.PathLike
            path to create the directory into
        theory : dict
            theory card to be dumped
        operator : dict
            operator card to be dumped
        metadata : dict
            metadata of the operator

        """
        dirpath = pathlib.Path(dirpath)

        # upgrade metadata
        # TODO: why?
        metadata_to_dump = copy.deepcopy(metadata)
        metadata_to_dump["rotations"]["xgrid"] = copy.deepcopy(
            operator["rotations"]["xgrid"]
        )
        metadata_to_dump["rotations"]["pids"] = br.flavor_basis_pids
        # upgrade operator
        # TODO: why?
        operator_to_dump = copy.deepcopy(operator)
        # keep only q2 grid
        operator_to_dump["Q2grid"] = np.array(
            list(operator_to_dump["Q2grid"].keys())
            if isinstance(operator_to_dump["Q2grid"], dict)
            else operator_to_dump["Q2grid"]
        )
        # TODO: this looks like some operations are being done twice and undone
        operator_obj = OperatorCard.from_dict(operator_to_dump).raw
        metadata_obj = {}
        metadata_obj["rotations"] = Rotations.from_dict(
            metadata_to_dump["rotations"]
        ).raw

        # write objects
        paths = InternalPaths(dirpath)
        paths.metadata.write_text(yaml.dump(metadata_obj), encoding="utf-8")
        paths.theory_card.write_text(yaml.dump(theory), encoding="utf-8")
        paths.operator_card.write_text(yaml.dump(operator_obj), encoding="utf-8")
        paths.recipes.mkdir()
        paths.parts.mkdir()
        paths.operators.mkdir()

    @property
    def metadata(self) -> dict:
        """Provide metadata, retrieving from the dump."""
        return yaml.safe_load(self.paths.metadata.read_text(encoding="utf-8"))

    @property
    def theory(self) -> dict:
        """Provide theory card, retrieving from the dump."""
        # TODO: make an actual attribute, move load to `theory_card`, and the
        # type of the attribute will be `eko.runcards.TheoryCard`
        return yaml.safe_load(self.paths.theory_card.read_text(encoding="utf-8"))

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

    def update_metadata(self, to_update: dict):
        """Update the metadata file with the info in to_update."""
        # cast to update to list
        for attr in to_update["rotations"]:
            to_update["rotations"][attr] = to_update["rotations"][attr].tolist()
        new_metadata = copy.deepcopy(self.metadata)
        new_metadata["rotations"].update(to_update["rotations"])
        new_metadata_string = yaml.safe_dump(new_metadata).encode()
        stream = io.BytesIO(new_metadata_string)
        stream.seek(0)
        info = tarfile.TarInfo(name="metadata.yaml")
        info.size = len(stream.getbuffer())
        info.mtime = int(time.time())
        info.mode = 436
        with tarfile.open(self.path, "a") as tar:
            tar.addfile(info, fileobj=stream)

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
    def detached(cls, theory: dict, operator: dict, metadata: dict, path: pathlib.Path):
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
        metadata: dict
            the metadata card
        path : os.PathLike
            the underlying path (it has to be a valid object, but it is not
            guaranteed, see the note)

        Returns
        -------
        EKO
            the generated structure

        """
        bases = copy.deepcopy(metadata["rotations"])
        # cast bases list to numpy array
        for attr in bases:
            bases[attr] = np.array(bases[attr])
        bases["pids"] = np.array(br.flavor_basis_pids)
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
        metadata: dict
            metadata of the operator
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
        # Constructing initial metadata
        metadata = dict(rotations=dict())
        metadata["rotations"]["_targetgrid"] = copy.deepcopy(
            operator["rotations"]["xgrid"]
        )
        metadata["rotations"]["_inputgrid"] = copy.deepcopy(
            operator["rotations"]["xgrid"]
        )
        metadata["rotations"]["_targetpids"] = np.array(
            copy.deepcopy(operator["rotations"]["pids"])
        )
        metadata["rotations"]["_inputpids"] = np.array(
            copy.deepcopy(operator["rotations"]["pids"])
        )
        for bases in ["inputgrid", "targetgrid", "inputpids", "targetpids"]:
            if bases in operator["rotations"]:
                del operator["rotations"][bases]
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            cls.bootstrap(td, theory=theory, operator=operator, metadata=metadata)

            with tarfile.open(path, mode="w") as tar:
                for element in td.glob("*"):
                    tar.add(element, arcname=element.name)
            shutil.rmtree(td)
        eko = cls.detached(theory, operator, metadata, path=path)
        logger.info(f"New operator created at path '{path}'")
        return eko

    @classmethod
    def load(cls, path: os.PathLike):
        """Load dump into an :class:`EKO` object.

        Parameters
        ----------
        path : os.PathLike
            path to the dump to load

        Returns
        -------
        EKO
            the loaded instance

        """
        path = pathlib.Path(path)
        if not tarfile.is_tarfile(path):
            raise exceptions.OutputNotTar(f"Not a valid tar archive: '{path}'")

        theory = yaml.safe_load(cls.extract(path, THEORYFILE))
        operator = yaml.safe_load(cls.extract(path, OPERATORFILE))
        metadata = yaml.safe_load(cls.extract(path, METADATAFILE))

        eko = cls.detached(theory, operator, metadata, path=path)
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
            Q0=float(np.sqrt(self.Q02)),
            Q2grid=self.mu2grid.tolist(),
            rotations=self.rotations.raw,
        )

    def __del__(self):
        """Destroy the memory structure gracefully."""
        self.unload()
