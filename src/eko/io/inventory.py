"""Manage assets used during computation."""
import base64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

from .access import AccessConfigs
from .items import Header, Operator

NBYTES = 8
ENDIANNES = "little"

HEADER_EXT = ".yaml"
ARRAY_EXT = [".npy", ".npz"]
COMPRESSED_EXT = ".lz4"
OPERATOR_EXT = [ext + COMPRESSED_EXT for ext in ARRAY_EXT]


class LookupError(ValueError):
    """Failure in content retrieval from inventory."""


def encode(header: Header):
    """Extract an hash from a header."""
    return base64.urlsafe_b64encode(
        hash(header).to_bytes(NBYTES, byteorder=ENDIANNES)
    ).decode(encoding="utf-8")


def header_name(header: Header):
    """Determine header file name."""
    stem = encode(header)
    return stem + HEADER_EXT


def operator_name(header: Header, err: bool):
    """Determine operator file name, from the associated header."""
    stem = encode(header)
    return stem + OPERATOR_EXT[1 if err else 0]


@dataclass(frozen=True)
class Inventory:
    """Assets manager.

    In particular, manage autosave, autoload, and memory caching.

    """

    access: AccessConfigs
    cache: Dict[Header, Optional[Operator]] = field(default_factory=dict)
    contentful: bool = True

    def lookup(self, stem: str) -> Path:
        """Look up for content path in inventory."""
        found = [
            path
            for path in self.access.path.iterdir()
            if path.name.startswith(stem)
            if "".join(path.suffixes) in OPERATOR_EXT
        ]

        if len(found) == 0:
            raise LookupError(f"Target value '{stem}' not available.")
        elif len(found) > 1:
            raise LookupError(f"Too many operators associated to '{stem}':\n{found}")

        return found[0]

    def __getitem__(self, header: Header) -> Optional[Operator]:
        r"""Retrieve operator for given header.

        If the operator is not already in memory, it will be automatically
        loaded.

        """
        self.access.assert_open()

        if header in self.cache:
            op = self.cache[header]
            if op is not None or not self.contentful:
                return op

        stem = encode(header)
        oppath = self.lookup(stem)

        with open(oppath, "rb") as fd:
            op = Operator.load(fd)

        self.cache[header] = op
        return op

    def __setitem__(self, header: Header, operator: Optional[Operator]):
        """Set operator for given :math:`mu^2`.

        Header and operator are automatically dumped on disk.

        """
        self.access.assert_writeable()

        if not self.contentful:
            self.cache[header] = None

        assert operator is not None

        headpath = self.access.path / header_name(header)
        with_err = operator.error is not None
        oppath = self.access.path / operator_name(header, err=with_err)

        headpath.write_text(yaml.dump(asdict(header)))
        with open(oppath, "wb") as fd:
            operator.save(fd)

        self.cache[header] = operator

    def __delitem__(self, header: Header):
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
