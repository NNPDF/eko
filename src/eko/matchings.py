r"""Holds the classes that define the |FNS|."""

import logging
from dataclasses import dataclass
from typing import List, Union

import numba as nb
import numpy as np

from .constants import MTAU
from .io.types import EvolutionPoint as EPoint
from .io.types import FlavorIndex, FlavorsNumber, SquaredScale
from .quantities.heavy_quarks import MatchingScales

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    """Oriented path in the threshold landscape."""

    origin: SquaredScale
    """Starting point."""
    target: SquaredScale
    """Final point."""
    nf: FlavorsNumber
    """Number of active flavors."""

    @property
    def is_downward(self) -> bool:
        """Return True if ``origin`` is bigger than ``target``."""
        return self.origin > self.target

    def __str__(self):
        """Textual representation, mainly for logging purpose."""
        return f"Segment({self.origin} -> {self.target}, nf={self.nf})"


Path = List[Segment]


@dataclass(frozen=True)
class Matching:
    """Matching between two different segments.

    The meaning of the flavor index `hq` is the PID of the corresponding heavy
    quark.
    """

    scale: SquaredScale
    hq: FlavorIndex
    inverse: bool


MatchedPath = List[Union[Segment, Matching]]


class Atlas:
    r"""Holds information about the matching scales.

    These scales are the :math:`Q^2` has to pass in order to get there
    from a given :math:`Q^2_{ref}`.
    """

    def __init__(self, matching_scales: MatchingScales, origin: EPoint):
        """Create basic atlas."""
        self.walls = [0] + matching_scales + [np.inf]
        self.origin = self.normalize(origin)

        logger.info(str(self))

    def normalize(self, target: EPoint) -> EPoint:
        """Fill number of flavors if needed."""
        if target[1] is not None:
            return target
        return (target[0], nf_default(target[0], self))

    def __str__(self):
        """Textual representation, mainly for logging purpose."""
        walls = " - ".join([f"{w:.2e}" for w in self.walls])
        return f"Atlas [{walls}], ref={self.origin[0]} @ {self.origin[1]}"

    @classmethod
    def ffns(cls, nf: int, mu2: SquaredScale):
        """Create a |FFNS| setup.

        The function creates simply sufficient thresholds at ``0`` (in the
        beginning), since the number of flavors is determined by counting
        from below.

        The origin is set with that number of flavors.
        """
        matching_scales = MatchingScales([0] * (nf - 3) + [np.inf] * (6 - nf))
        origin = (mu2, nf)
        return cls(matching_scales, origin)

    def path(self, target: EPoint) -> Path:
        """Determine the path to the target evolution point.

        Essentially, the path is always monotonic in the number of flavors,
        increasing or decreasing the active flavors by one unit every time a
        matching happens at the suitable scale.

        Examples
        --------
        Since this can result in a counter-intuitive behavior, let's walk through some examples.

        Starting with the intuitive one:
        >>> Atlas([10, 20, 30], (5, 3)).path((25, 5))
        [Segment(5, 10, 3), Segment(10, 20, 4), Segment(20, 25, 5)]

        If the number of flavor has been reached, it will continue walking
        without matchin again.
        >>> Atlas([10, 20, 30], (5, 3)).path((25, 4))
        [Segment(5, 10, 3), Segment(10, 25, 4)]

        It is irrelevant the scale you start from, to step from 3 to 4 you have
        to cross the charm matching scale, whether this means walking upward or
        downward.
        >>> Atlas([10, 20, 30], (15, 3)).path((25, 5))
        [Segment(15, 10, 3), Segment(10, 20, 4), Segment(20, 25, 5)]

        An actual backward evolution is defined by lowering the number of
        flavors going from origin to target.
        >>> Atlas([10, 20, 30], (25, 5)).path((5, 3))
        [Segment(25, 20, 5), Segment(20, 10, 4), Segment(10, 5, 3)]

        But the only difference is in the matching between two segments, since
        a single segment is always happening in a fixed number of flavors, and
        it is completely analogue for upward or downward evolution.

        Note
        ----

        Since the only task required to determine a path is interleaving the
        correct matching scales, this is done by slicing the walls.
        """
        mu20, nf0 = self.origin
        mu2f, nff = self.normalize(target)

        # determine direction and python slice modifier
        rc, shift = (-1, -3) if nff < nf0 else (1, -2)

        # join all necessary points in one list
        boundaries = [mu20] + self.walls[nf0 + shift : nff + shift : rc] + [mu2f]

        return [
            Segment(boundaries[i], mu2, nf0 + i * rc)
            for i, mu2 in enumerate(boundaries[1:])
        ]

    def matched_path(self, target: EPoint) -> MatchedPath:
        """Determine the path to the target, including matchings.

        In practice, just a wrapper around :meth:`path` adding the intermediate
        matchings.
        """
        path = self.path(target)
        inverse = is_downward_path(path)

        prev = path[0]
        matched: MatchedPath = [prev]
        for seg in path[1:]:
            matching = Matching(prev.target, max(prev.nf, seg.nf), inverse)

            matched.append(matching)
            matched.append(seg)
            prev = seg

        return matched


def nf_default(mu2: SquaredScale, atlas: Atlas) -> FlavorsNumber:
    r"""Determine the number of active flavors in the *default flow*.

    Default flow is defined by the natural sorting of the matching scales:

    .. math::

        \mu_c < \mu_b < \mu_t

    So, the flow is defined starting with 3 flavors below the charm matching,
    and increasing by one every time a matching scale is passed while
    increasing the scale.
    """
    ref_idx = np.digitize(mu2, atlas.walls)
    return int(2 + ref_idx)


def is_downward_path(path: Path) -> bool:
    r"""Determine if a path is downward.

    Criterias are:

    - in the number of active flavors when the path list contains more than one
      :class:`Segment`, note this can be different from each
      :attr:`Segment.is_downward`
    - in :math:`\mu^2`, when just one single :class:`Segment` is given
    """
    if len(path) == 1:
        return path[0].is_downward
    return path[1].nf < path[0].nf


def flavor_shift(is_downward: bool) -> int:
    """Determine the shift to number of light flavors."""
    return 4 if is_downward else 3


@nb.njit(cache=True)
def lepton_number(q2):
    """Compute the number of leptons.

    Note: muons and electrons are always massless as for up, down and strange.

    Parameters
    ----------
    q2 : float
        scale

    Returns
    -------
    int :
       Number of leptons
    """
    return 3 if q2 > MTAU**2 else 2
