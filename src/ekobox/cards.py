"""Tools to generate runcards."""
import copy
import os
from typing import Any, Optional

import yaml
from banana.data import sql, theories

from eko import basis_rotation as br
from ekomark.data import operators

Card = dict[str, Any]


def generate_operator(
    Q2grid: list[float],
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
    def_op = copy.deepcopy(operators.default_card)
    # Adding the mandatory inputs
    def_op["pids"] = list(br.flavor_basis_pids)
    def_op["Q2grid"] = Q2grid
    if isinstance(update, dict):
        for k in update.keys():
            if k not in def_op.keys():
                raise ValueError("Provided key not in operators card")
        for key, value in update.items():
            def_op[key] = value
    serialized = sql.serialize(def_op)
    def_op["hash"] = (sql.add_hash(serialized))[-1]
    if path is not None:
        dump(path, def_op)
    return def_op


def generate_theory(
    pto: int,
    initial_scale: float,
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
    theory = copy.deepcopy(theories.default_card)
    # Adding the mandatory inputs
    theory["PTO"] = pto
    theory["Q0"] = initial_scale
    # Update user choice
    if update is not None:
        for k in update.keys():
            if k not in theory.keys():
                raise ValueError("Provided key not in theory card")
        theory.update(update)
    serialized = sql.serialize(theory)
    theory["hash"] = sql.add_hash(serialized)[-1]
    if path is not None:
        dump(path, theory)
    return theory


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
