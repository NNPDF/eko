"""Define output representation structures."""

import contextlib
import copy
import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from .. import interpolation
from . import exceptions, raw, v1, v2
from .access import AccessConfigs
from .inventory import Inventory
from .items import Evolution, Matching, Operator, Recipe, Target
from .metadata import Metadata
from .paths import InternalPaths
from .runcards import OperatorCard, TheoryCard
from .types import EvolutionPoint as EPoint
from .types import SquaredScale

logger = logging.getLogger(__name__)

TEMP_PREFIX = "eko-"


def inventories(path: Path, access: AccessConfigs) -> dict:
    """Set up empty inventories for object initialization."""
    paths = InternalPaths(path)
    return dict(
        recipes=Inventory(
            paths.recipes, access, Evolution, contentless=True, name="recipes"
        ),
        recipes_matching=Inventory(
            paths.recipes_matching,
            access,
            Matching,
            contentless=True,
            name="matching-recipes",
        ),
        parts=Inventory(paths.parts, access, Evolution, name="parts"),
        parts_matching=Inventory(
            paths.parts_matching, access, Matching, name="matching-parts"
        ),
        operators=Inventory(paths.operators, access, Target, name="operators"),
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

    recipes: Inventory[Evolution]
    recipes_matching: Inventory[Matching]
    parts: Inventory[Evolution]
    parts_matching: Inventory[Matching]
    operators: Inventory[Target]

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
    def xgrid(self) -> interpolation.XGrid:
        """Momentum fraction internal grid."""
        return self.metadata.xgrid

    @xgrid.setter
    def xgrid(self, value: interpolation.XGrid):
        """Set `xgrid` value."""
        self.metadata.xgrid = value
        self.update()

    @property
    def mu20(self) -> SquaredScale:
        """Provide squared initial scale."""
        return self.metadata.origin[0]

    @property
    def mu2grid(self) -> List[SquaredScale]:
        """Provide the list of :math:`Q^2` as an array."""
        return [ep.scale for ep in self.operators]

    @property
    def evolgrid(self) -> List[EPoint]:
        """Provide the list of evolution points as an array."""
        return list(self)

    @property
    def theory_card(self):
        """Provide theory card, retrieving from the dump."""
        raw_th = yaml.safe_load(self.paths.theory_card.read_text(encoding="utf-8"))
        if self.metadata.data_version in [1]:
            raw_th = v1.update_theory(raw_th)
        if self.metadata.data_version in [2]:
            raw_th = v2.update_theory(raw_th)
        return TheoryCard.from_dict(raw_th)

    @property
    def operator_card(self):
        """Provide operator card, retrieving from the dump."""
        raw_op = yaml.safe_load(self.paths.operator_card.read_text(encoding="utf-8"))
        if self.metadata.data_version in [1]:
            # here we need to read also the theory card
            raw_th = yaml.safe_load(self.paths.theory_card.read_text(encoding="utf-8"))
            raw_op = v1.update_operator(raw_op, raw_th)
        if self.metadata.data_version in [2]:
            # here we need to read also the theory card
            raw_th = yaml.safe_load(self.paths.theory_card.read_text(encoding="utf-8"))
            raw_op = v2.update_operator(raw_op, raw_th)
        return OperatorCard.from_dict(raw_op)

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

    # recipes management
    # -------------------

    def load_recipes(self, recipes: List[Recipe]):
        """Load recipes in bulk."""
        for recipe in recipes:
            # leverage auto-save
            if isinstance(recipe, Evolution):
                self.recipes[recipe] = None
            else:
                self.recipes_matching[recipe] = None

    # operator management
    # -------------------
    def __getitem__(self, ep: EPoint) -> Optional[Operator]:
        r"""Retrieve operator for given evolution point."""
        return self.operators[Target.from_ep(ep)]

    def __setitem__(self, ep: EPoint, op: Operator):
        """Set operator associated to an evolution point."""
        self.operators[Target.from_ep(ep)] = op

    def __delitem__(self, ep: EPoint):
        """Drop operator from memory."""
        del self.operators[Target.from_ep(ep)]

    def __delattr__(self, name: str):
        """Empty an inventory cache."""
        attr = getattr(self, name)
        if isinstance(attr, Inventory):
            attr.empty()
        else:
            super().__delattr__(name)

    @contextlib.contextmanager
    def operator(self, ep: EPoint):
        """Retrieve an operator and discard it afterwards.

        To be used as a contextmanager: the operator is automatically loaded as
        usual, but on the closing of the context manager it is dropped from
        memory.
        """
        try:
            yield self[ep]
        finally:
            del self[ep]

    def __iter__(self):
        """Iterate over keys (i.e. evolution points)."""
        for target in self.operators:
            yield target.ep

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
        for target in self.operators:
            # recast to evolution point
            ep = target.ep

            # auto-load
            op = self[ep]
            assert op is not None
            yield ep, op
            # auto-unload
            del self[ep]

    def __contains__(self, ep: EPoint) -> bool:
        """Check whether the operator related to the evolution point is
        present.

        'Present' means, in this case, they are available in the :class:`EKO`.
        But it is telling nothing about being loaded in memory or not.
        """
        return Target.from_ep(ep) in self.operators

    def approx(
        self, ep: EPoint, rtol: float = 1e-6, atol: float = 1e-10
    ) -> Optional[EPoint]:
        r"""Look for close enough evolution point in the :class:`EKO`.

        The distance is mostly evaluated along the :math:`\mu^2` dimension,
        while :math:`n_f` is considered with a discrete distance: if two points
        have not the same, they are classified as far.

        Raises
        ------
        ValueError
            if multiple values are found in the neighbourhood
        """
        eps = np.array([ep_ for ep_ in self if ep_[1] == ep[1]])
        mu2s = np.array([mu2 for mu2, _ in eps])
        close = eps[np.isclose(ep[0], mu2s, rtol=rtol, atol=atol)]

        if len(close) == 1:
            found = close[0]
            assert isinstance(found[0], float)
            return (found[0], int(found[1]))
        if len(close) == 0:
            return None
        raise ValueError(f"Multiple values of Q2 have been found close to {ep}")

    def unload(self):
        """Fully unload the operators in memory."""
        for ep in self:
            del self[ep]

    # operator management
    # -------------------

    def deepcopy(self, path: Path):
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
        new.access.path = path
        new.access.readonly = False
        new.access.open = True

        tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
        new.metadata.path = tmpdir
        # copy old dir to new dir
        tmpdir.rmdir()
        shutil.copytree(self.paths.root, new.paths.root)
        new.close()

    @classmethod
    def load(cls, path: Path):
        """Load the EKO from disk information.

        Note
        ----
        No archive path is assigned to the :class:`EKO` object, setting its
        :attr:`EKO.access.path` to `None`.
        If you want to properly load from an archive, use the :meth:`read`
        constructor.
        """
        access = AccessConfigs(None, readonly=True, open=True)

        metadata = Metadata.load(path)
        loaded = cls(
            **inventories(path, access),
            metadata=metadata,
            access=access,
        )
        loaded.operators.sync()

        # check operator shape is still compatible
        for _, op in loaded.items():
            try:
                assert op.operator.shape[1] == op.operator.shape[3]
                assert op.operator.shape[0] == op.operator.shape[2]
            except AssertionError as exc:
                raise ValueError(
                    "Loading not squared EKOs is no longer possible."
                ) from exc

        return loaded

    @classmethod
    def read(
        cls,
        path: Path,
        extract: bool = True,
        dest: Optional[Path] = None,
        readonly: bool = True,
    ):
        """Load an existing EKO.

        If the `extract` attribute is `True` the EKO is loaded from its archived
        format. Otherwise, the `path` is interpreted as the location of an
        already extracted folder.
        """
        # Take the absolute path in case we need to modify the eko in-place
        path = path.resolve()
        if extract:
            dir_ = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX)) if dest is None else dest
            with tarfile.open(path) as tar:
                raw.safe_extractall(tar, dir_)
        else:
            dir_ = path

        loaded = cls.load(dir_)

        loaded.access.readonly = readonly
        if extract:
            loaded.access.path = path

        return loaded

    @classmethod
    def create(cls, path: Path):
        """Create a new EKO."""
        access = AccessConfigs(path, readonly=False, open=True)
        builder = Builder(
            path=Path(tempfile.mkdtemp(prefix=TEMP_PREFIX)), access=access
        )
        return builder

    @classmethod
    def edit(cls, *args, **kwargs):
        """Read from and write on existing EKO.

        Alias of `EKO.read(..., readonly=False)`, see :meth:`read`.
        """
        return cls.read(*args, readonly=False, **kwargs)

    def __enter__(self):
        """Allow EKO to be used in :obj:`with` statements."""
        return self

    def dump(self, archive: Optional[Path] = None):
        """Dump the current content to archive.

        Parameters
        ----------
        archive: Path or None
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

        If not in read-only mode, dump to permanent storage. Remove the
        temporary directory used.
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

        if self.access.path is not None:
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
        return dict(mu2grid=self.mu2grid, metadata=self.metadata.raw)


@dataclass
class Builder:
    """Build EKO instances."""

    path: Path
    """Path on disk to the EKO."""
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
            origin=(self.operator.init[0] ** 2, self.operator.init[1]),
            xgrid=self.operator.xgrid,
        )
        InternalPaths(self.path).bootstrap(
            theory=self.theory.raw,
            operator=self.operator.raw,
            metadata=metadata.raw,
        )

        self.eko = EKO(
            **inventories(self.path, self.access),
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
