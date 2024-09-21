"""Manage assets used during computation."""

import base64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Generic, Literal, Optional, Type, TypeVar

import yaml

from .access import AccessConfigs
from .items import Header, Operator

NBYTES = 8
ENDIANNESS: Literal["little", "big"] = "little"

HEADER_EXT = ".yaml"
ARRAY_EXT = [".npy", ".npz"]
COMPRESSED_EXT = ".lz4"
OPERATOR_EXT = [ext + COMPRESSED_EXT for ext in ARRAY_EXT]


class LookupError(ValueError):
    """Failure in content retrieval from inventory."""


def encode(header: Header):
    """Extract an hash from a header."""
    return base64.urlsafe_b64encode(
        abs(hash(header)).to_bytes(NBYTES, byteorder=ENDIANNESS)
    ).decode(encoding="utf-8")


def header_name(header: Header):
    """Determine header file name."""
    stem = encode(header)
    return stem + HEADER_EXT


def operator_name(header: Header, err: bool):
    """Determine operator file name, from the associated header."""
    stem = encode(header)
    return stem + OPERATOR_EXT[1 if err else 0]


H = TypeVar("H", bound=Header)


@dataclass(frozen=True)
class Inventory(Generic[H]):
    """Assets manager.

    In particular, manage autosave, autoload, and memory caching.
    """

    path: Path
    access: AccessConfigs
    header_type: Type[Header]
    cache: Dict[H, Optional[Operator]] = field(default_factory=dict)
    contentless: bool = False
    name: Optional[str] = None
    """Only for logging purpose."""

    def __str__(self) -> str:
        return f"Inventory '{self.name}'"

    def lookup(self, stem: str, header: bool = False) -> Path:
        """Look up for content path in inventory."""
        EXT = OPERATOR_EXT if not header else [HEADER_EXT]
        found = [
            path
            for path in self.path.iterdir()
            if path.name.startswith(stem)
            if "".join(path.suffixes) in EXT
        ]

        if len(found) == 0:
            raise LookupError(f"Item '{stem}' not available in {self}.")
        elif len(found) > 1:
            raise LookupError(
                f"Too many items associated to '{stem}' in {self}:\n{found}"
            )

        return found[0]

    def __getitem__(self, header: H) -> Optional[Operator]:
        r"""Retrieve operator for given header.

        If the operator is not already in memory, it will be
        automatically loaded.
        """
        self.access.assert_open()

        try:
            op = self.cache[header]
            if op is not None or self.contentless:
                return op
        except KeyError:
            pass

        stem = encode(header)

        # in case of contentless, check header availability instead
        if self.contentless:
            self.lookup(stem, header=True)
            self.cache[header] = None
            return None

        # for contentful inventories, check operator availability
        oppath = self.lookup(stem)

        with open(oppath, "rb") as fd:
            op = Operator.load(fd)

        self.cache[header] = op
        return op

    def __setitem__(self, header: H, operator: Optional[Operator]):
        """Set operator for given header.

        Header and operator are automatically dumped on disk.
        """
        self.access.assert_writeable()

        # always save the header on disk
        headpath = self.path / header_name(header)
        headpath.write_text(yaml.dump(asdict(header)), encoding="utf-8")

        # in case of contentless, set empty cache and exit
        if self.contentless:
            self.cache[header] = None
            return

        # otherwise save also the operator, and add to the cache
        assert operator is not None

        with_err = operator.error is not None
        oppath = self.path / operator_name(header, err=with_err)

        with open(oppath, "wb") as fd:
            operator.save(fd)

        self.cache[header] = operator

    def __delitem__(self, header: H):
        """Drop operator from memory.

        Irrelevant for contentless inventories.

        Note
        ----
        This method only drops the operator from memory, and it's not expected
        to do anything else.

        Autosave is done on set, and explicit saves are performed by the
        computation functions.

        If a further explicit save is required, repeat explicit assignment::

            inventory[header] = inventory[header]

        This is only useful if the operator has been mutated in place, that in
        general should be avoided, since the operator should only be the result
        of a full computation or a library manipulation.
        """
        self.cache[header] = None

    def __iter__(self):
        """Iterate over loaded content.

        This iteration is only over cache, so it might not be faithful with
        respect to the real content on disk.
        To iterate the full content of the disk, just call right before
        :meth:`sync`.
        """
        yield from self.cache

    def __len__(self):
        """Return the number of elements in the cache."""
        return len(self.cache)

    def sync(self):
        """Sync the headers in the cache with the content on disk.

        In particular, headers on disk that are missing in the :attr:`cache`
        are added to it, without loading actual operators in memory.

        Despite the name, the operation is non-destructive, so, even if cache
        has been abused, nothing will be deleted nor unloaded.
        """
        for path in self.path.iterdir():
            if path.suffix != HEADER_EXT:
                continue

            header = self.header_type(
                **yaml.safe_load(path.read_text(encoding="utf-8"))
            )
            self.cache[header] = None

    def __invert__(self):
        """Alias for :meth:`sync`."""
        self.sync()

    def empty(self):
        """Empty the in-memory cache."""
        for header in self.cache:
            del self[header]
