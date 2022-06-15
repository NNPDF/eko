# -*- coding: utf-8 -*-
import pytest

from eko import compatibility

theory1 = {
    "alphas": 0.1180,
    "alphaqed": 0.007496,
    "PTO": 2,
    "QED": 0,
}


def test_compatibility():
    new_theory = compatibility.update(theory1)


operator_dict = {"ev_op_max_order": 2}


def test_compatibility_operators():
    new_theory, new_operator = compatibility.update(theory1, operator_dict)


theory2 = {
    "alphas": 0.1180,
    "alphaqed": 0.007496,
    "PTO": 2,
}


def test_compatibility_no_QED():
    new_theory = compatibility.update(theory2)


theory3 = {
    "alphas": 0.1180,
    "alphaqed": 0.007496,
    "QED": 0,
}

with pytest.raises(KeyError):
    new_theory = compatibility.update(theory3)
