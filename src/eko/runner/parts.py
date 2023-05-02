"""Compute operator components.

.. todo::

    This is the only part of the new runner making use of the global context
    through the :class:`EKO` object.

    After #242, the goal is to update :class:`Operator` and
    :class:`OperatorMatrixElement` to simplify the computation and passing down
    parameters.

"""
import numpy as np

from .. import EKO
from .. import evolution_operator as evop
from ..evolution_operator import operator_matrix_element as ome
from ..io.items import Evolution, Matching, Operator
from ..quantities.heavy_quarks import QuarkMassScheme
from . import commons


def managers(eko: EKO) -> dict:
    """Collect managers for operator computation.

    .. todo::

        Legacy interface, avoid managers usage.

    """
    tcard = eko.theory_card
    ocard = eko.operator_card
    return dict(
        thresholds_config=commons.atlas(tcard, ocard),
        couplings=commons.couplings(tcard, ocard),
        interpol_dispatcher=commons.interpolator(ocard),
    )


def evolve_configs(eko: EKO) -> dict:
    """Create configs for :class:`Operator`.

    .. todo::

        Legacy interface, make use of a dedicated object.

    """
    # self.config:
    # - order
    # - n_integration_cores
    # - xif2
    # - ev_op_iterations
    # - debug_skip_non_singlet
    # - debug_skip_singlet
    # - method
    # - ev_op_max_order
    # - polarized
    # - time_like
    tcard = eko.theory_card
    ocard = eko.operator_card
    return dict(
        order=tcard.order,
        xif2=tcard.xif**2,
        method=ocard.configs.evolution_method,
        ev_op_iterations=ocard.configs.ev_op_iterations,
        ev_op_max_order=ocard.configs.ev_op_max_order,
        polarized=ocard.configs.polarized,
        time_like=ocard.configs.time_like,
        debug_skip_singlet=ocard.debug.skip_singlet,
        debug_skip_non_singlet=ocard.debug.skip_non_singlet,
        n_integration_cores=ocard.configs.n_integration_cores,
    )


def evolve(eko: EKO, recipe: Evolution) -> Operator:
    """Compute evolution in isolation."""
    op = evop.Operator(
        evolve_configs(eko), managers(eko), recipe.as_atlas, is_threshold=recipe.cliff
    )
    op.compute()
    return Operator(np.array([]))


def matching_configs(eko: EKO) -> dict:
    """Create configs for :class:`OperatorMatrixElement`.

    .. todo::

        Legacy interface, make use of a dedicated object.

    """
    # self.config:
    # - order
    # - n_integration_cores
    # - xif2
    # - ev_op_iterations
    # - debug_skip_non_singlet
    # - debug_skip_singlet
    # - method
    # - ev_op_max_order
    # - polarized
    # - time_like
    # - backward_inversion
    # - intrinsic_range
    tcard = eko.theory_card
    ocard = eko.operator_card
    return dict(
        order=tcard.order,
        xif2=tcard.xif**2,
        method=ocard.configs.evolution_method,
        ev_op_iterations=ocard.configs.ev_op_iterations,
        ev_op_max_order=ocard.configs.ev_op_max_order,
        polarized=ocard.configs.polarized,
        time_like=ocard.configs.time_like,
        debug_skip_singlet=ocard.debug.skip_singlet,
        debug_skip_non_singlet=ocard.debug.skip_non_singlet,
        n_integration_cores=ocard.configs.n_integration_cores,
        backward_inversion=ocard.configs.inversion_method,
        intrinsic_range=tcard.heavy.intrinsic_flavors,
    )


def match(eko: EKO, recipe: Matching) -> Operator:
    """Compute matching in isolation."""
    kthr = eko.theory_card.heavy.squared_ratios[recipe.hq - 4]
    op = ome.OperatorMatrixElement(
        matching_configs(eko),
        managers(eko),
        recipe.hq,
        recipe.scale,
        recipe.inverse,
        np.log(kthr),
        eko.theory_card.heavy.masses_scheme is QuarkMassScheme.MSBAR,
    )
    op.compute()
    return Operator(np.array([]))
