"""Compute operator components.

.. todo::

    This is the only part of the new runner making use of the global context
    through the :class:`EKO` object.

    After #242, the goal is to update :class:`Operator` and
    :class:`OperatorMatrixElement` to simplify the computation and passing down
    parameters.
"""

import numpy as np

from .. import evolution_operator as evop
from ..evolution_operator import matching_condition, physical
from ..evolution_operator import operator_matrix_element as ome
from ..io import EKO
from ..io.items import Evolution, Matching, Operator
from ..quantities.heavy_quarks import QuarkMassScheme
from . import commons


def _managers(eko: EKO) -> evop.Managers:
    """Collect managers for operator computation.

    .. todo::

        Legacy interface, avoid managers usage.
    """
    tcard = eko.theory_card
    ocard = eko.operator_card
    return evop.Managers(
        atlas=commons.atlas(tcard, ocard),
        couplings=commons.couplings(tcard, ocard),
        interpolator=commons.interpolator(ocard),
    )


def _evolve_configs(eko: EKO) -> dict:
    """Create configs for :class:`Operator`.

    .. todo::

        Legacy interface, make use of a dedicated object.
    """
    tcard = eko.theory_card
    ocard = eko.operator_card
    return dict(
        order=tcard.order,
        xif2=tcard.xif**2,
        method=ocard.configs.evolution_method.value,
        ev_op_iterations=ocard.configs.ev_op_iterations,
        ev_op_max_order=ocard.configs.ev_op_max_order,
        polarized=ocard.configs.polarized,
        time_like=ocard.configs.time_like,
        debug_skip_singlet=ocard.debug.skip_singlet,
        debug_skip_non_singlet=ocard.debug.skip_non_singlet,
        n_integration_cores=ocard.configs.n_integration_cores,
        ModSV=ocard.configs.scvar_method,
        n3lo_ad_variation=tcard.n3lo_ad_variation,
        use_fhmruvv=tcard.use_fhmruvv,
        # Here order is shifted by one, no QED matching is available so far.
        matching_order=tcard.matching_order,
    )


def evolve(eko: EKO, recipe: Evolution) -> Operator:
    """Compute evolution in isolation."""
    op = evop.Operator(
        _evolve_configs(eko),
        _managers(eko),
        recipe.as_atlas,
        is_threshold=recipe.cliff,
    )
    op.compute()

    qed = eko.theory_card.order[1] > 0
    res, err = physical.PhysicalOperator.ad_to_evol_map(
        op.op_members, op.nf, op.q2_to, qed
    ).to_flavor_basis_tensor(qed)

    return Operator(res, err)


def _matching_configs(eko: EKO) -> dict:
    """Create configs for :class:`OperatorMatrixElement`.

    .. todo::

        Legacy interface, make use of a dedicated object.
    """
    ocard = eko.operator_card
    return dict(
        **_evolve_configs(eko),
        backward_inversion=ocard.configs.inversion_method,
    )


def match(eko: EKO, recipe: Matching) -> Operator:
    """Compute matching in isolation.

    Note
    ----
    Compared to the old prescription, a dedicated rotation to a different
    intrinsic basis is not needed any longer.

    All the operators are blown up to flavor basis, and they are saved and
    joined in that unique basis. So, the only rotation used is towards that
    basis, and encoded in the blowing up prescription.
    """
    kthr = eko.theory_card.heavy.squared_ratios[recipe.hq - 4]
    op = ome.OperatorMatrixElement(
        _matching_configs(eko),
        _managers(eko),
        recipe.hq - 1,
        recipe.scale,
        recipe.inverse,
        np.log(kthr),
        eko.theory_card.heavy.masses_scheme is QuarkMassScheme.MSBAR,
    )
    op.compute()
    qed = eko.theory_card.order[1] > 0
    res, err = matching_condition.MatchingCondition.split_ad_to_evol_map(
        op.op_members, op.nf, recipe.scale, qed
    ).to_flavor_basis_tensor(qed)

    return Operator(res, err)
