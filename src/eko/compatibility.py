# -*- coding: utf-8 -*-
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
    else:
        if "PTO" in new_theory:
            new_theory["order"] = (new_theory.pop("PTO") + 1, 0)
    if "ev_op_max_order" in new_operators:
        if isinstance(new_operators["ev_op_max_order"], int):
            new_operators["ev_op_max_order"] = (
                new_operators["ev_op_max_order"],
                new_theory["order"][1],
            )
    return new_theory, new_operators
