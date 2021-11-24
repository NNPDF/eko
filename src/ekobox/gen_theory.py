import io
import math
import pathlib

import banana
import yaml
from banana.data import sql


def gen_theory_card(pto, initial_scale, update=None, export=False, name="MyTheoryCard"):
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
        export : bool
            set if dump
        name : str
            name of theory card (if exported )

    Returns
    -------

        : dict
        theory card
    """
    # Constructing the dictionary with some default value
    here = pathlib.Path(banana.__file__).parent / "data"
    with open(here / "theory_template.yaml", "r") as o:
        theory = yaml.safe_load(o)
    # delete unuseful member
    del theory["FNS"]
    # Adding the mandatory inputs
    theory["PTO"] = pto
    theory["Q0"] = initial_scale
    serialized = sql.serialize(theory)
    theory["hash"] = (sql.add_hash(serialized))[-1]
    # Update user choice
    if isinstance(update, dict):
        for k in update.keys():
            if k not in theory.keys():
                raise ValueError("Provided key not in theory card")
        theory.update(update)
    if export:
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
    target = "%s.yaml" % (name)
    with open(target, "w") as out:
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
    with open(path, "r") as o:
        theory = yaml.safe_load(o)
    return theory
