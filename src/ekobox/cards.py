"""Tools to generate runcards."""
import os
from math import nan
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from eko import basis_rotation as br
from eko.io import runcards

Card = Dict[str, Any]

_theory = dict(
    order=[1, 0],
    couplings=dict(alphas=[0.118, 91.2], alphaem=[0.007496252, nan]),
    num_flavs_ref=None,
    num_flavs_init=None,
    num_flavs_max_as=6,
    num_flavs_max_pdf=6,
    intrinsic_flavors=[4],
    quark_masses=[2.0, 4.5, 173.07],
    quark_masses_scheme="POLE",
    matching=[1.0, 1.0, 1.0],
    fact_to_ren=1.0,
)

_operator = dict(
    mu0=1.65,
    _mugrid=[100.0],
    configs=dict(
        evolution_method="iterate-exact",
        ev_op_max_order=[10, 0],
        ev_op_iterations=10,
        interpolation_polynomial_degree=4,
        interpolation_is_log=True,
        scvar_method=None,
        inversion_method=None,
        n_integration_cores=0,
    ),
    debug=dict(
        skip_singlet=False,
        skip_non_singlet=False,
    ),
    rotations=dict(
        xgrid=np.geomspace(1e-7, 1.0, 50).tolist(),
        pids=list(br.flavor_basis_pids),
    ),
)


class example:
    """Provide runcards examples."""

    @classmethod
    def theory(cls) -> runcards.TheoryCard:
        """Provide example theory card object."""
        return runcards.TheoryCard.from_dict(_theory)

    @classmethod
    def operator(cls) -> runcards.OperatorCard:
        """Provide example operator card object."""
        return runcards.OperatorCard.from_dict(_operator)

    @classmethod
    def raw_theory(cls):
        """Provide example theory card unstructured."""
        # TODO: consider to return instead `cls.theory().raw`
        return _theory.copy()

    @classmethod
    def raw_operator(cls):
        """Provide example operator card unstructured."""
        return _theory.copy()


def update_card(card: runcards.Card, update: Card):
    """Update card with dictionary content."""
    for k, v in update.items():
        if not hasattr(card, k):
            raise ValueError(f"The key '{k}' is not in '{type(card)}'")
        setattr(card, k, v)


def generate_theory(
    pto: int,
    update: Optional[Card] = None,
    path: Optional[os.PathLike] = None,
) -> Card:
    """Generate theory card.

    Generates a theory card with some mandatory user choice and some
    default values which can be changed by the update input dict

    Parameters
    ----------
        pto : int
            perturbation theory order
        initial_scale: float
            initial scale of evolution [GeV]
        update : dict
            info to update to default theory card
        name : str
            name of exported theory card (if name is not None )

    Returns
    -------
        dict
            theory card
    """
    # Constructing the dictionary with some default values
    theory = example.theory()
    # Adding the mandatory inputs
    theory.order = (pto, theory.order[1])
    # Update user choice
    if update is not None:
        update_card(theory, update)

    raw = theory.raw
    if path is not None:
        dump(path, raw)
    return raw


def generate_operator(
    initial_scale: float,
    Q2grid: List[float],
    update: Optional[Card] = None,
    path: Optional[os.PathLike] = None,
) -> Card:
    """Generate operators card.

    Generates an operators card with some mandatory user choice
    (in this case only the Q2 grid) and some default values which
    can be changed by the update input dict.

    Parameters
    ----------
    Q2grid : list(float)
        grid for Q2
    update : dict
        dictionary of info to update in op. card
    path : os.PathLike
        name of exported op.card (if name not None)

    Returns
    -------
    dict
        operators card

    """
    # Constructing the dictionary with some default value
    operator = example.operator()
    # Adding the mandatory inputs
    operator.mu0 = initial_scale
    operator._mu2grid = np.array(Q2grid)
    operator.rotations.pids = np.array(br.flavor_basis_pids)
    # Update user choice
    if update is not None:
        update_card(operator, update)

    raw = operator.raw
    if path is not None:
        dump(path, raw)
    return raw


def dump(path: os.PathLike, card: Card):
    """Export the operators card.

    Parameters
    ----------
    name : str
        name of the operators card to export
    card : dict
        card to dump

    """
    with open(path, "w", encoding="utf-8") as fd:
        yaml.safe_dump(card, fd)


def load(path) -> Card:
    """Import the theory card specified by path.

    Parameters
    ----------
    path : str
        path to theory card in yaml format

    Returns
    -------
    dict
        loaded card

    """
    with open(path, encoding="utf-8") as fd:
        card = yaml.safe_load(fd)

    return card
