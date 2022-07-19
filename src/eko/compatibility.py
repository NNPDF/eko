# -*- coding: utf-8 -*-
"""Compatibility functions.

Upgrade old input (NNPDF jargon compatible) to the native one.

"""
import copy


def update(theory, operators):
    """
    Upgrade the legacy theory and observable runcards with the new settings.

    Parameters
    ----------
        theory : dict
            theory runcard
        observables : dict
            observable runcard

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
        if "configs" not in operators:
            raise ValueError("No subsections, old format.")

        max_order = new_operators["configs"]["ev_op_max_order"]
        if isinstance(max_order, int):
            new_operators["configs"]["ev_op_max_order"] = (
                max_order,
                new_theory["order"][1],
            )
    return new_theory, new_operators


def update_theory(theory):
    """
    Upgrade the legacy theory runcards with the new settings.

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
