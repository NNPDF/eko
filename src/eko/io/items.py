"""Inventory items definition."""

import io
from dataclasses import asdict, dataclass
from typing import BinaryIO, Optional, Union

import lz4.frame
import numpy as np
import numpy.typing as npt
from numpy.lib import npyio

from .. import matchings
from . import exceptions
from .types import EvolutionPoint as EPoint
from .types import FlavorIndex, FlavorsNumber, SquaredScale


@dataclass(frozen=True)
class Header:
    """Item header, containing metadata."""


@dataclass(frozen=True)
class Evolution(Header):
    """Information to compute an evolution operator.

    It describes the evolution with a fixed number of light flavors
    between two scales.
    """

    origin: SquaredScale
    """Starting point."""
    target: SquaredScale
    """Final point."""
    nf: FlavorsNumber
    """Number of active flavors."""

    cliff: bool = False
    """Whether the operator is reaching a matching scale.

    Cliff operators are the only ones allowed to be intermediate, even though
    they can also be final segments of an evolution path (see
    :meth:`eko.matchings.Atlas.path`).

    Intermediate ones always have final scales :attr:`mu2` corresponding to
    matching scales, and initial scales :attr:`mu20` corresponding to either
    matching scales or the global initial scale of the |EKO|.

    Note
    ----

    The name of *cliff* operators stems from the following diagram::

        nf = 3 --------------------------------------------------------
                        |
        nf = 4 --------------------------------------------------------
                                |
        nf = 5 --------------------------------------------------------
                                                            |
        nf = 6 --------------------------------------------------------

    where each lane corresponds to |DGLAP| evolution with the relative number
    of running flavors, and the vertical bridges are the perturbative matchings
    between two different "adjacent" schemes.
    """

    @classmethod
    def from_atlas(cls, segment: matchings.Segment, cliff: bool = False):
        """Create instance from analogous :class:`eko.matchings.Atlas`
        object."""
        return cls(**asdict(segment), cliff=cliff)

    @property
    def as_atlas(self) -> matchings.Segment:
        """The associated segment."""
        return matchings.Segment(self.origin, self.target, self.nf)


@dataclass(frozen=True)
class Matching(Header):
    """Information to compute a matching operator.

    Describe the matching between two different flavor number schemes.
    """

    scale: SquaredScale
    hq: FlavorIndex
    inverse: bool

    @classmethod
    def from_atlas(cls, matching: matchings.Matching):
        """Create instance from analogous :class:`eko.matchings.Atlas`
        object."""
        return cls(**asdict(matching))

    @property
    def as_atlas(self) -> matchings.Matching:
        """The associated segment."""
        return matchings.Matching(self.scale, self.hq, self.inverse)


Recipe = Union[Evolution, Matching]


@dataclass(frozen=True)
class Target(Header):
    """Target evolution point, labeling evolution from origin to there."""

    scale: SquaredScale
    nf: FlavorsNumber

    @classmethod
    def from_ep(cls, ep: EPoint):
        """Create instance from the :class:`EPoint` analogue."""
        return cls(*ep)

    @property
    def ep(self) -> EPoint:
        """Cast to :class:`EPoint`."""
        return (self.scale, self.nf)


@dataclass(frozen=True)
class Operator:
    """Operator representation.

    To be used to hold the result of a computed 4-dim operator (either a raw
    evolution operator or a matching condition).

    Note
    ----
    IO works with streams in memory, in order to avoid intermediate write on
    disk (keep read from and write to tar file only).
    """

    operator: npt.NDArray
    """Content of the evolution operator."""
    error: Optional[npt.NDArray] = None
    """Errors on individual operator elements (mainly used for integration
    error, but it can host any kind of error)."""

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
            # TODO: We might consider dropping this exception since np.load will always
            # return a array (or fail on it's own)
            raise exceptions.OperatorLoadingError(
                "Not possible to load operator, content format not recognized"
            )

        return cls(operator=op, error=err)


@dataclass(frozen=True)
class Item:
    """Inventory item."""

    header: Header
    content: Operator
