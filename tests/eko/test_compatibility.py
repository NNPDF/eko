# -*- coding: utf-8 -*-
from eko import compatibility

theory_dict = {
    "alphas": 0.1180,
    "alphaqed": 0.007496,
    "Qref": 91,
    "nfref": 5,
    "MaxNfPdf": 6,
    "MaxNfAs": 6,
    "Q0": 1,
    "fact_to_ren_scale_ratio": 1.0,
    "mc": 1.5,
    "mb": 4.1,
    "mt": 175.0,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "HQ": "MSBAR",
    "Qmc": 18,
    "Qmb": 20,
    "Qmt": 175.0,
    "PTO": 2,
    "QED": 0,
    "ModEv": "EXA",
}


def test_compatibility():
    new_theory = compatibility.update(theory_dict)


theory_dict = {
    "alphas": 0.1180,
    "alphaqed": 0.007496,
    "Qref": 91,
    "nfref": 5,
    "MaxNfPdf": 6,
    "MaxNfAs": 6,
    "Q0": 1,
    "fact_to_ren_scale_ratio": 1.0,
    "mc": 1.5,
    "mb": 4.1,
    "mt": 175.0,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "HQ": "MSBAR",
    "Qmc": 18,
    "Qmb": 20,
    "Qmt": 175.0,
    "PTO": 2,
    "ModEv": "EXA",
}


def test_compatibility_no_QED():
    new_theory = compatibility.update(theory_dict)


operator_dict = {"ev_op_max_order": 2}


def test_compatibility_operators():
    new_theory, new_operator = compatibility.update(theory_dict, operator_dict)
