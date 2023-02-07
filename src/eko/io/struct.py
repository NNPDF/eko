"""Define output representation structures."""
import base64
import contextlib
import copy
import io
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, Dict, Optional

import lz4.frame
import numpy as np
import numpy.lib.npyio as npyio
import numpy.typing as npt
import yaml

from .. import interpolation
from .. import version as vmod
from . import exceptions, raw
from .dictlike import DictLike
from .runcards import OperatorCard, Rotations, TheoryCard

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
        aux = io.BytesIO()
        if self.error is None:
            np.save(aux, self.operator)
        else:
            np.savez(aux, operator=self.operator, error=self.error)
        aux.seek(0)

        # compress if requested
        if compress:
            content = lz4.frame.compress(aux.read())
        else:
            content = aux.read()

        # write compressed data
        stream.write(content)
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
    with :attr:`root` equal to emtpty string or ``"."``.

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

    def bootstrap(self, theory: dict, operator: dict, metadata: dict):
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
        self.metadata.write_text(yaml.dump(metadata), encoding="utf-8")
        self.theory_card.write_text(yaml.dump(theory), encoding="utf-8")
        self.operator_card.write_text(yaml.dump(operator), encoding="utf-8")
        self.recipes.mkdir()
        self.parts.mkdir()
        self.operators.mkdir()

    @staticmethod
    def opname(mu2: float) -> str:
        r"""Operator file name from :math:`\mu^2` value."""
        decoded = np.float64(mu2).tobytes()
        return base64.urlsafe_b64encode(decoded).decode()

    def oppath(self, mu2: float) -> pathlib.Path:
        r"""Retrieve operator file path from :math:`\mu^2` value.

        This method looks for an existing path matching.

        Parameters
        ----------
        mu2: float
            :math:`\mu2` scale specified

        Returns
        -------
        pathlib.Path
            the path retrieved, guaranteed to exist

        Raises
        ------
        ValueError
            if the path is not found, or more than one are matching the
            specified value of :math:`\mu2`

        """
        name = self.opname(mu2)
        oppaths = list(
            filter(lambda path: path.name.startswith(name), self.operators.iterdir())
        )

        if len(oppaths) == 0:
            raise ValueError(f"mu2 value '{mu2}' not available.")
        elif len(oppaths) > 1:
            raise ValueError(f"Too many operators associated to '{mu2}':\n{oppaths}")

        return oppaths[0]

    def opcompressed(self, path: os.PathLike) -> bool:
        """Check if the operator at the path specified is compressed.

        Parameters
        ----------
        path: os.PathLike
            the path to the operator to check

        Returns
        -------
        bool
            whether it is compressed

        Raises
        ------
        OperatorLocationError
            if the path is not inside the operators folder

        """
        path = pathlib.Path(path)
        if self.operators not in path.parents:
            raise exceptions.OperatorLocationError(path)

        return path.suffix == COMPRESSED_SUFFIX

    def opmu2(self, path: os.PathLike) -> float:
        r"""Extract :math:`\mu2` value from operator path.

        Parameters
        ----------
        path: os.PathLike
            the path to the operator

        Returns
        -------
        bool
            the :math:`\mu2` value associated to the operator

        Raises
        ------
        OperatorLocationError
            if the path is not inside the operators folder

        """
        path = pathlib.Path(path)
        if self.operators not in path.parents:
            raise exceptions.OperatorLocationError(path)

        encoded = path.stem.split(".")[0]
        decoded = base64.urlsafe_b64decode(encoded)
        return float(np.frombuffer(decoded, dtype=np.float64)[0])

    def opnewpath(
        self, mu2: float, compress: bool = True, without_err: bool = True
    ) -> pathlib.Path:
        r"""Compute the path associated to :math:`\mu^2` value.

        Parameters
        ----------
        mu2: float
            :math:`\mu2` scale specified

        Returns
        -------
        pathlib.Path
            the path computed, it might already exists

        """
        suffix = ".npy" if without_err else ".npz"
        if compress:
            suffix += COMPRESSED_SUFFIX
        return self.operators / (self.opname(mu2) + suffix)

    @property
    def mu2grid(self) -> npt.NDArray[np.float64]:
        r"""Provide the array of :math:`\mu2` values of existing operators."""
        if self.root is None:
            raise RuntimeError()

        return np.array(
            sorted(
                map(
                    lambda p: self.opmu2(p),
                    InternalPaths(self.root).operators.iterdir(),
                )
            )
        )


@dataclass
class AccessConfigs:
    """Configurations specified during opening of an EKO."""

    path: pathlib.Path
    """The path to the permanent object."""
    readonly: bool
    "Read-only flag"
    open: bool
    "EKO status"

    def assert_open(self):
        """Assert operator is open.

        Raises
        ------
        exceptions.ClosedOperator
            if operator is closed

        """
        if not self.open:
            raise exceptions.ClosedOperator

    def assert_writeable(self, msg: Optional[str] = None):
        """Assert operator is writeable.

        Raises
        ------
        exceptions.ClosedOperator
            see :meth:`assert_open`
        exceptions.ReadOnlyOperator
            if operators has been declared read-only

        """
        if msg is None:
            msg = ""

        self.assert_open()
        if self.readonly:
            raise exceptions.ReadOnlyOperator(msg)

    @property
    def read(self):
        """Check reading permission.

        Reading access is always granted on open operator.

        """
        return self.open

    @property
    def write(self):
        """Check writing permission."""
        return self.open and not self.readonly


@dataclass
class Metadata(DictLike):
    """Manage metadata, and keep them synced on disk.

    It is possible to have a metadata view, in which the path is not actually
    connected (i.e. it is set to ``None``). In this case, no update will be
    possible, of course.

    Note
    ----
    Unfortunately, for nested structures it is not possible to detect a change
    in their attributes, so a call to :meth:`update` has to be performed
    manually.

    """

    mu20: float
    """Inital scale."""
    rotations: Rotations
    """Manipulation information, describing the current status of the EKO (e.g.
    `inputgrid` and `targetgrid`).
    """
    # tagging information
    _path: Optional[pathlib.Path] = None
    """Path to temporary dir."""
    version: str = vmod.__version__
    """Library version used to create the corresponding file."""
    data_version: int = vmod.__data_version__
    """Specs version, to which the file adheres."""

    @classmethod
    def load(cls, path: os.PathLike):
        """Load metadata from open folder.

        Parameters
        ----------
        path: os.PathLike
            the path to the open EKO folder

        Returns
        -------
        bool
            loaded metadata

        """
        path = pathlib.Path(path)
        content = cls.from_dict(
            yaml.safe_load(InternalPaths(path).metadata.read_text(encoding="utf-8"))
        )
        content._path = path
        return content

    def update(self):
        """Update the disk copy of metadata."""
        if self._path is None:
            logger.info("Impossible to set metadata, no file attached.")
        else:
            with open(InternalPaths(self._path).metadata, "w") as fd:
                yaml.safe_dump(self.raw, fd)

    @property
    def path(self):
        """Access temporary dir path.

        Raises
        ------
        RuntimeError
            if path has not been initialized before

        """
        if self._path is None:
            raise RuntimeError(
                "Access to EKO directory attempted, but not dir has been set."
            )
        return self._path

    @path.setter
    def path(self, value: pathlib.Path):
        """Set temporary dir path."""
        self._path = value

    @property
    def raw(self):
        """Override default :meth:`DictLike.raw` representation to exclude path."""
        raw = super().raw

        for key in raw.copy():
            if key.startswith("_"):
                del raw[key]

        return raw


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

    # public containers
    # -----------------
    metadata: Metadata
    """Operator metadata."""
    access: AccessConfigs
    """Access related configurations."""

    # shortcut properties
    # -------------------
    @property
    def paths(self) -> InternalPaths:
        """Accessor for internal paths."""
        return InternalPaths(self.metadata.path)

    @property
    def rotations(self) -> Rotations:
        """Rotations information."""
        return self.metadata.rotations

    @property
    def xgrid(self) -> interpolation.XGrid:
        """Momentum fraction internal grid."""
        return self.rotations.xgrid

    @xgrid.setter
    def xgrid(self, value: interpolation.XGrid):
        """Set `xgrid` value."""
        self.rotations.xgrid = value
        self.update()

    @property
    def mu20(self) -> float:
        """Provide squared initial scale."""
        return self.metadata.mu20

    @property
    def mu2grid(self) -> npt.NDArray:
        """Provide the list of :math:`Q^2` as an array."""
        return np.array(list(self._operators))

    @property
    def theory_card(self):
        """Provide theory card, retrieving from the dump."""
        return TheoryCard.from_dict(
            yaml.safe_load(self.paths.theory_card.read_text(encoding="utf-8"))
        )

    @property
    def operator_card(self):
        """Provide operator card, retrieving from the dump."""
        return OperatorCard.from_dict(
            yaml.safe_load(self.paths.operator_card.read_text(encoding="utf-8"))
        )

    # persistency control
    # -------------------

    def update(self):
        """Write updates to structure for persistency."""
        self.access.assert_writeable()
        self.metadata.update()

    def assert_permissions(self, read=True, write=False):
        """Assert permissions on current operator."""
        if read:
            self.access.assert_open()
        if write:
            self.access.assert_writeable()

    @property
    def permissions(self):
        """Provide permissions information."""
        return dict(read=self.access.read, write=self.access.write)

    # operator management
    # -------------------
    def __getitem__(self, mu2: float) -> Operator:
        r"""Retrieve operator for given :math:`\mu^2`.

        If the operator is not already in memory, it will be automatically
        loaded.

        Parameters
        ----------
        mu2 : float
            :math:`\mu^2` value labeling the operator to be retrieved

        Returns
        -------
        Operator
            the retrieved operator

        """
        self.access.assert_open()

        if mu2 in self._operators:
            op = self._operators[mu2]
            if op is not None:
                return op

        oppath = self.paths.oppath(mu2)
        compressed = self.paths.opcompressed(oppath)

        with open(oppath, "rb") as fd:
            op = Operator.load(fd, compressed=compressed)

        self._operators[mu2] = op
        return op

    def __setitem__(self, mu2: float, op: Operator, compress: bool = True):
        """Set operator for given :math:`Q^2`.

        The operator is automatically dumped on disk.

        Parameters
        ----------
        q2 : float
            :math:`Q^2` value labeling the operator to be set
        op : Operator
            the retrieved operator
        compress : bool
            whether to save the operator compressed or not (default: `True`)

        """
        self.access.assert_writeable()

        if not isinstance(op, Operator):
            raise ValueError("Only an Operator can be added to an EKO")

        without_err = op.error is None
        oppath = self.paths.opnewpath(mu2, compress=compress, without_err=without_err)

        with open(oppath, "wb") as fd:
            without_err2 = op.save(fd, compress)
        assert without_err == without_err2

        self._operators[mu2] = op

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

    # operator management
    # -------------------

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
        path :
            path to the permanent location of the new object (not the temporary
            directory)

        Returns
        -------
        EKO
            the copy created

        """
        # deepcopy is tuned to copy an open operator into a closed one, since
        # operators are always created closed (to avoid delegating the user to
        # close an object he never opened), then let's assert this assumption
        self.access.assert_open()

        # return the object fully on disk, in order to avoid expensive copies
        # also in memory (they would be copied on-disk in any case)
        self.unload()

        new = copy.deepcopy(self)
        new.access.path = pathlib.Path(path)
        new.access.readonly = False
        new.access.open = True

        tmpdir = pathlib.Path(tempfile.mkdtemp())
        new.metadata.path = tmpdir
        # copy old dir to new dir
        tmpdir.rmdir()
        shutil.copytree(self.paths.root, new.paths.root)
        new.close()

    @staticmethod
    def load(tarpath: os.PathLike, dest: os.PathLike):
        """Load the content of archive in a target directory.

        Parameters
        ----------
        tarpath: os.PathLike
            the archive to extract
        tmppath: os.PathLike
            the destination directory

        """
        try:
            with tarfile.open(tarpath) as tar:
                raw.safe_extractall(tar, dest)
        except tarfile.ReadError:
            raise exceptions.OutputNotTar(f"Not a valid tar archive: '{tarpath}'")

    @classmethod
    def open(cls, path: os.PathLike, mode="r"):
        """Open EKO object in the specified mode."""
        path = pathlib.Path(path)
        access = AccessConfigs(path, readonly=False, open=True)
        load = False
        if mode == "r":
            load = True
            access.readonly = True
        elif mode in "w":
            pass
        elif mode in "a":
            load = True
        else:
            raise ValueError

        tmpdir = pathlib.Path(tempfile.mkdtemp())
        if load:
            cls.load(path, tmpdir)
            metadata = Metadata.load(tmpdir)
            opened = cls(
                _operators={mu2: None for mu2 in InternalPaths(metadata.path).mu2grid},
                metadata=metadata,
                access=access,
            )
        else:
            opened = Builder(path=tmpdir, access=access)

        return opened

    @classmethod
    def read(cls, path: os.PathLike):
        """Read the content of an EKO.

        Type-safe alias for::

            EKO.open(... , "r")

        """
        eko = cls.open(path, "r")
        assert isinstance(eko, EKO)
        return eko

    @classmethod
    def create(cls, path: os.PathLike):
        """Create a new EKO.

        Type-safe alias for::

            EKO.open(... , "w")

        """
        builder = cls.open(path, "w")
        assert isinstance(builder, Builder)
        return builder

    @classmethod
    def edit(cls, path: os.PathLike):
        """Read from and write on existing EKO.

        Type-safe alias for::

            EKO.open(... , "a")

        """
        eko = cls.open(path, "a")
        assert isinstance(eko, EKO)
        return eko

    def __enter__(self):
        """Allow EKO to be used in :obj:`with` statements."""
        return self

    def dump(self, archive: Optional[os.PathLike] = None):
        """Dump the current content to archive.

        Parameters
        ----------
        archive: os.PathLike or None
            path to archive, in general you should keep the default, that will
            make use of the registered path (default: ``None``)

        Raises
        ------
        ValueError
            when trying to dump on default archive in read-only mode

        """
        if archive is None:
            self.access.assert_writeable(
                "Not possible to dump on default archive in read-only mode"
            )
            archive = self.access.path

        with tarfile.open(archive, "w") as tar:
            tar.add(self.metadata.path, arcname=".")

    def close(self):
        """Close the current object, cleaning up.

        If not in read-only mode, dump to permanent storage.
        Remove the temporary directory used.

        """
        if not self.access.readonly:
            # clean given path, to overwrite it - default 'w'rite behavior
            self.access.path.unlink(missing_ok=True)
            self.dump()

        self.access.open = False
        shutil.rmtree(self.metadata.path)

    def __exit__(self, exc_type: type, _exc_value, _traceback):
        """Ensure EKO to be closed properly."""
        if exc_type is not None:
            return

        self.close()

    @property
    def raw(self) -> dict:
        """Provide raw representation of the full content.

        Returns
        -------
        dict
            nested dictionary, storing all the values in the structure, but the
            operators themselves

        """
        return dict(Q2grid=self.mu2grid.tolist(), metadata=self.metadata.raw)


@dataclass
class Builder:
    """Build EKO instances."""

    path: pathlib.Path
    """Path on disk to ."""
    access: AccessConfigs
    """Access related configurations."""

    # optional arguments, required at build time
    theory: Optional[TheoryCard] = None
    operator: Optional[OperatorCard] = None

    eko: Optional[EKO] = None

    def __post_init__(self):
        """Validate paths."""
        if self.access.path.suffix != ".tar":
            raise exceptions.OutputNotTar(self.access.path)
        if self.access.path.exists():
            raise exceptions.OutputExistsError(self.access.path)

    def load_cards(self, theory: TheoryCard, operator: OperatorCard):
        """Load both theory and operator card."""
        self.theory = theory
        self.operator = operator

        return self

    def build(self) -> EKO:
        """Build EKO instance.

        Returns
        -------
        EKO
            the constructed instance

        Raises
        ------
        RuntimeError
            if not enough information is available (at least one card missing)

        """
        missing = []
        for card in ["theory", "operator"]:
            if getattr(self, card) is None:
                missing.append(card)

        if len(missing) > 0:
            raise RuntimeError(
                f"Can not build an EKO, since following cards are missing: {missing}"
            )

        # tell the static analyzer as well
        assert self.theory is not None
        assert self.operator is not None

        self.access.open = True
        metadata = Metadata(
            _path=self.path,
            mu20=self.operator.mu20,
            rotations=copy.deepcopy(self.operator.rotations),
        )
        InternalPaths(self.path).bootstrap(
            theory=self.theory.raw,
            operator=self.operator.raw,
            metadata=metadata.raw,
        )

        self.eko = EKO(
            _operators={mu2: None for mu2 in self.operator.mu2grid},
            metadata=metadata,
            access=self.access,
        )

        return self.eko

    def __enter__(self):
        """Allow Builder to be used in :obj:`with` statements."""
        return self

    def __exit__(self, exc_type: type, _exc_value, _traceback):
        """Ensure EKO to be closed properly."""
        if exc_type is not None:
            return

        # assign to variable to help type checker, otherwise self.eko might be
        # a property, and its type can change at every evaluation
        eko = self.eko
        if eko is not None:
            eko.close()
