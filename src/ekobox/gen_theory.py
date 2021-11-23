import io
import math
import pathlib

import yaml
from banana.data import sql


def gen_theory_card(fns, pto, initial_scale, update=None):
    """
    Generates a theory card with some mandatory user choice and some
    default values which can be changed by the update input dict
    """
    # Constructing the dictionary with some default value (NB: ask if it relies on order)
    def_theory = {
        "CKM": "0.97428 0.22530 0.003470 0.22520 0.97345 0.041000 0.00862 0.04030 0.999152",
        "Comments": "",
        "DAMP": 0,
        "EScaleVar": 1,
        #    'FNS': 'FFNS',
        "GF": 1.1663787e-05,
        "HQ": "POLE",
        "IB": 0,
        "IC": 0,
        "ID": 0,
        "MP": 0.938,
        "MW": 80.398,
        "MZ": 91.1876,
        "MaxNfAs": 6,
        "MaxNfPdf": 6,
        "ModEv": "EXA",
        "NfFF": 3,
        #    'PTO': 0,
        #    'Q0': 1.0,
        "QED": 0,
        "Qedref": 1.777,
        "Qmb": 4.5,
        "Qmc": 2.0,
        "Qmt": 173.07,
        "Qref": 91.2,
        "SIN2TW": 0.23126,
        "SxOrd": "LL",
        "SxRes": 0,
        "TMC": 0,
        "XIF": 1.0,
        "XIR": 1.0,
        "alphaqed": 0.007496251999999999,
        "alphas": 0.11800000000000001,
        "fact_to_ren_scale_ratio": 1.0,
        "global_nx": 0,
        #    'hash': 'edfbed6',
        "kDISbThr": 1.0,
        "kDIScThr": 1.0,
        "kDIStThr": 1.0,
        "kbThr": 1.0,
        "kcThr": 1.0,
        "ktThr": 1.0,
        "mb": 4.5,
        "mc": 2.0,
        "mt": 173.07,
        "nf0": None,
        "nfref": None,
    }
    # Adding the mandatory inputs
    def_theory["FNS"] = fns
    def_theory["PTO"] = pto
    def_theory["Q0"] = initial_scale
    serialized = sql.serialize(def_theory)
    def_theory["hash"] = (sql.add_hash(serialized))[-1]
    # Update user choice
    if isinstance(update, dict):
        for k in update.keys():
            if k not in def_theory.keys():
                raise ValueError("Provided key not in theory card")
        def_theory.update(update)
    return def_theory


def dump_theory_card(name, theory):
    """
    Dump the theory card in the current directory

    Parameters
    ----------
        name : str
            name of the theory card to dump

        theory : dict
            theory card
    """
    target = "theorycard_%s.yaml" % (name)
    with open(target, "w") as out:
        yaml.safe_dump(theory, out)


def load_theory_card(path):
    """
    Load the theory card specified by path

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


def create_info_file(theory_card, operators_card, info_update):
    """
    Generate a lhapdf info file from theory and operators card

    Parameters
    ----------
        theory_card : dict
            theory card
        operators_card : dict
            operators_card
        info_update : dict
            info to update

    Returns
    -------
        : dict
        info file in lhapdf format
    """
    here = pathlib.Path(__file__).parent
    with open(here / "templatePDF.info", "r") as o:
        template_info = yaml.safe_load(o)
    # TODOs: Updating the info using the theory and operator cards
    template_info.update(info_update)
    return template_info
