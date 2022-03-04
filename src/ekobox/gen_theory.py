import copy

import yaml
from banana.data import sql, theories


def gen_theory_card(pto, initial_scale, update=None, name=None):
    """
    Generates a theory card with some mandatory user choice and some
    default values which can be changed by the update input dict

    Parameters
    ----------

        pto : int
            perturbation theory order
        initial_scale: float
            initial scale of evolution
        update : dict
            info to update to default theory card
        name : str
            name of exported theory card (if name not None )

    Returns
    -------

        : dict
        theory card
    """
    # Constructing the dictionary with some default values
    theory = copy.deepcopy(theories.default_card)
    # delete unuseful member
    del theory["FNS"]
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
    theory["hash"] = (sql.add_hash(serialized))[-1]
    if name is not None:
        export_theory_card(name, theory)
    return theory


def export_theory_card(name, theory):
    """
    Export the theory card in the current directory

    Parameters
    ----------
        name : str
            name of the theory card to export

        theory : dict
            theory card
    """
    target = f"{name}.yaml"
    with open(target, "w", encoding="utf-8") as out:
        yaml.safe_dump(theory, out)


def import_theory_card(path):
    """
    Import the theory card specified by path

    Parameters
    ----------
        path : str
            path to theory card in yaml format

    Returns
    -------
        : dict
            theory card
    """
    with open(path, "r", encoding="utf-8") as o:
        theory = yaml.safe_load(o)
    return theory
