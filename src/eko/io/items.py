"""Inventory items definition."""
import io
from dataclasses import dataclass
from typing import BinaryIO, Optional

import lz4.frame
import numpy as np
import numpy.lib.npyio as npyio
import numpy.typing as npt

from . import exceptions


@dataclass(frozen=True)
class Header:
    """Item header, containing metadata."""


@dataclass(frozen=True)
class Operator:
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

    def save(self, stream: BinaryIO) -> bool:
        """Save content of operator to bytes.

        The content is saved on a `stream`, in order to be able to perform the
        operation both on disk and in memory.

        The returned value tells whether the operator saved contained or not
        the error (this control even the format, ``npz`` with errors, ``npy``
        otherwise).

        """
        aux = io.BytesIO()
        if self.error is None:
            np.save(aux, self.operator)
        else:
            np.savez(aux, operator=self.operator, error=self.error)
        aux.seek(0)

        # compress if requested
        content = lz4.frame.compress(aux.read())

        # write compressed data
        stream.write(content)
        stream.seek(0)

        # return the type of array dumped (i.e. 'npy' or 'npz')
        return self.error is None

    @classmethod
    def load(cls, stream: BinaryIO):
        """Load operator from bytes.

        An input `stream` is used to load the operator from, in order to
        support the operation both on disk and in memory.

        """
        extracted_stream = io.BytesIO(lz4.frame.decompress(stream.read()))
        content = np.load(extracted_stream)

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


@dataclass(frozen=True)
class Recipe(Header):
    """Information to compute an evolution operator."""

    matching: bool


@dataclass(frozen=True)
class RecipeMatching(Header):
    """Information to compute a matching operator."""

    matching: bool


@dataclass(frozen=True)
class Part(Header):
    """Evolution operator."""

    matching: bool


@dataclass(frozen=True)
class PartMatching(Header):
    """Matching operator."""

    matching: bool


@dataclass(frozen=True)
class Item:
    """Inventory item."""

    header: Header
    content: Operator
