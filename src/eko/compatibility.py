# -*- coding: utf-8 -*-


def update(theory):
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
    new_theory = theory.copy()
    # new_obs = observables.copy()
    new_theory["alphaem"] = new_theory.pop("alphaqed")
    new_theory["orders"] = (new_theory["PTO"], new_theory["QED"])
    del new_theory["PTO"]
    del new_theory["QED"]
    return new_theory
