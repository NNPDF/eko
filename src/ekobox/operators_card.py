# -*- coding: utf-8 -*-
"""Tools to generate operator cards."""
import copy

import yaml
from banana.data import sql

from ekomark.data import operators


def generate(Q2grid, update=None, name=None):
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
    name : str
        name of exported op.card (if name not None)

    Returns
    -------
    dict
        operators card
    """
    # Constructing the dictionary with some default value
    def_op = copy.deepcopy(operators.default_card)
    # Adding the mandatory inputs
    def_op["Q2grid"] = Q2grid
    if isinstance(update, dict):
        for k in update.keys():
            if k not in def_op.keys():
                raise ValueError("Provided key not in operators card")
        for key, value in update.items():
            def_op[key] = value
    serialized = sql.serialize(def_op)
    def_op["hash"] = (sql.add_hash(serialized))[-1]
    if name is not None:
        dump(name, def_op)
    return def_op


def dump(name, op):
    """Export the operators card.

    Parameters
    ----------
        name : str
            name of the operators card to export
        op : dict
            operators card
    """
    target = f"{name}.yaml"
    with open(target, "w", encoding="utf-8") as out:
        yaml.safe_dump(op, out)


def load(path):
    """Import the operators card.

    Parameters
    ----------
    path : str
        path to operators card in yaml format

    Returns
    -------
    dict
        operators card
    """
    with open(path, "r", encoding="utf-8") as o:
        op = yaml.safe_load(o)
    return op
