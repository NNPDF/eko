import math


def gen_theory_card(fns, pto, initial_scale, hash, update=None):
    """
    Generates a theory card with some mandatory user choice and some
    default values which can be changed by the update input dict
    """
    # Constructing the dictionary with some default value (NB: ask if it relies on order)
    def_theory = {
        "CKM": "0.97428 0.22530 0.003470 0.22520 "
        "0.97345 0.041000 0.00862 0.04030 "
        "0.999152",
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
        #    'Q0': 1.65,
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
        "kbThr": math.inf,
        "kcThr": math.inf,
        "ktThr": math.inf,
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
    def_theory["hash"] = hash
    # Update user choice (NB: Allow also new entries?)
    if isinstance(update, dict):
        for k in update.keys():
            if k not in def_theory.keys():
                raise ValueError("Provided key not in theory card")
        def_theory.update(update)
    return def_theory
