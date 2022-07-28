# -*- coding: utf-8 -*-
"""Compatibility functions.

Upgrade old input (NNPDF jargon compatible) to the native one.

"""
import copy
from typing import Optional


def update(theory: dict, operators: Optional[dict]):
    """Upgrade the legacy theory and observable runcards with the new settings.

    Parameters
    ----------
    theory : dict
        theory runcard
    observables : dict or None
        observable runcard (default: `None`)

    Returns
    -------
    new_theory : dict
        upgraded theory runcard
    new_obs : dict
        upgraded observable runcard

    """
    new_theory = copy.deepcopy(theory)
    new_operators = copy.deepcopy(operators)

    if "alphaqed" in new_theory:
        new_theory["alphaem"] = new_theory.pop("alphaqed")
    if "QED" in new_theory:
        new_theory["order"] = (new_theory.pop("PTO") + 1, new_theory.pop("QED"))

    if operators is not None:
        if new_operators is None:
            raise ValueError("Unreachable.")
        if "configs" not in operators:
            raise ValueError("No subsections, old format.")

        max_order = new_operators["configs"]["ev_op_max_order"]
        if isinstance(max_order, int):
            new_operators["configs"]["ev_op_max_order"] = (
                max_order,
                new_theory["order"][1],
            )

        new_operators["rotations"]["xgrid"] = operators["xgrid"]
        for basis in ("inputgrid", "targetgrid", "inputpids", "targetpids"):
            new_operators["rotations"][f"_{basis}"] = operators["rotations"][basis]

    return new_theory, new_operators


def update_theory(theory: dict):
    """Upgrade the legacy theory runcards with the new settings.

    Parameters
    ----------
    theory : dict
        theory runcard

    Returns
    -------
    new_theory : dict
        upgraded theory runcard

    """
    return update(theory, None)[0]
