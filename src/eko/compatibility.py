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

    if operators is not None and "configs" not in operators:
        assert new_operators is not None

        new_operators["configs"] = {}
        for k in (
            "interpolation_polynomial_degree",
            "interpolation_is_log",
            "ev_op_iterations",
            "backward_inversion",
            "n_integration_cores",
        ):
            new_operators["configs"][k] = operators[k]
        new_operators["rotations"] = {}
        new_operators["debug"] = {}
        for k in ("debug_skip_non_singlet", "debug_skip_singlet"):
            new_operators["debug"][k[len("debug_") :]] = operators[k]

        max_order = operators["ev_op_max_order"]
        if isinstance(max_order, int):
            new_operators["configs"]["ev_op_max_order"] = (
                max_order,
                new_theory["order"][1],
            )

        new_operators["rotations"]["xgrid"] = operators["interpolation_xgrid"]
        for basis in ("inputgrid", "targetgrid", "inputpids", "targetpids"):
            new_operators["rotations"][f"_{basis}"] = operators[basis]

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
