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
from ..evolution_operator import matching_condition
from ..evolution_operator import operator_matrix_element as ome
from ..evolution_operator import physical
from ..io import EKO
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


def blowup_info(eko: EKO) -> dict:
    """Prepare common information to blow up to flavor basis.

    Note
    ----
    ``intrinsic_range`` is a fully deprecated feature, here and anywhere else,
    since a full range is already always used for backward evolution, and it is
    not harmful to use it also for forward.

    Indeed, the only feature of non-intrinsic evolution is to absorb a
    non-trivial boundary condition when an intrinsic PDF is defined.
    But to achieve this, is sufficient to not specify any intrinsic boundary
    condition at all, while if something is there, it is intuitive enough that
    it will be consistently evolved.

    Moreover, since two different behavior are applied for the forward and
    backward evolution, the intrinsic range is a "non-local" function, since it
    does not depend only on the evolution segment, but also on the previous
    evolution history (to determine if evolution is backward in flavor,
    irrespectively of happening for an increasing or decreasing interval in
    scale at fixed flavor).

    """
    return dict(intrinsic_range=[4, 5, 6], qed=eko.theory_card.order[1] > 0)


def evolve_configs(eko: EKO) -> dict:
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
        matching_order=tcard.matching_order,
    )


def evolve(eko: EKO, recipe: Evolution) -> Operator:
    """Compute evolution in isolation."""
    op = evop.Operator(
        evolve_configs(eko), managers(eko), recipe.as_atlas, is_threshold=recipe.cliff
    )
    op.compute()

    binfo = blowup_info(eko)
    res, err = physical.PhysicalOperator.ad_to_evol_map(
        op.op_members, op.nf, op.q2_to, **binfo
    ).to_flavor_basis_tensor(qed=binfo["qed"])

    return Operator(res, err)


def matching_configs(eko: EKO) -> dict:
    """Create configs for :class:`OperatorMatrixElement`.

    .. todo::

        Legacy interface, make use of a dedicated object.

    """
    tcard = eko.theory_card
    ocard = eko.operator_card
    return dict(
        **evolve_configs(eko),
        backward_inversion=ocard.configs.inversion_method,
        intrinsic_range=tcard.heavy.intrinsic_flavors,
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
        matching_configs(eko),
        managers(eko),
        recipe.hq - 1,
        recipe.scale,
        recipe.inverse,
        np.log(kthr),
        eko.theory_card.heavy.masses_scheme is QuarkMassScheme.MSBAR,
    )
    op.compute()

    binfo = blowup_info(eko)
    res, err = matching_condition.MatchingCondition.split_ad_to_evol_map(
        op.op_members, op.nf, recipe.scale, **binfo
    ).to_flavor_basis_tensor(qed=binfo["qed"])

    return Operator(res, err)
