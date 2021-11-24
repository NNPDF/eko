import io

import yaml
from banana.data import sql
from ekomark.data import operators


def gen_op_card(Q2grid, update=None):
    """
    Generates an operator card with some mandatory user choice
    (in this case only the Q2 grid) and some default values which
    can be changed by the update input dict

    Parameters
    ----------
        Q2grid : list(float)
            grid for Q2
        update : dict
            dictionary of info to update in op. card

    Returns
    -------
        : dict
            operator card
    """
    # Constructing the dictionary with some default value
    def_op = operators.default_card
    # Adding the mandatory inputs
    def_op["Q2grid"] = Q2grid
    serialized = sql.serialize(def_op)
    def_op["hash"] = (sql.add_hash(serialized))[-1]
    if isinstance(update, dict):
        for k in update.keys():
            if k not in def_op.keys():
                raise ValueError("Provided key not in operators card")
        def_op.update(update)
    return def_op


def export_op_card(name, op):
    """
    Dump the operators card in the current directory

    Parameters
    ----------
        name : str
            name of the op. card to dump

        op : dict
            op card
    """
    target = "%s.yaml" % (name)
    with open(target, "w") as out:
        yaml.safe_dump(op, out)


def import_op_card(path):
    """
    Load the operators card specified by path

    Parameters
    ----------
        path : str
            path to op. card in yaml format

    Returns
    -------
        : dict
            op card
    """
    with open(path, "r") as o:
        op = yaml.safe_load(o)
    return op
