# -*- coding: utf-8 -*-
"""Transform the theory_card in order to be compatible with EKO."""
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
    if "alphaem_running" not in new_theory:
        new_theory["alphaem_running"] = False
        # TODO : add alphaem_running to the runcard
    if operators is not None:
        if isinstance(new_operators["ev_op_max_order"], int):
            new_operators["ev_op_max_order"] = (
                new_operators["ev_op_max_order"],
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
