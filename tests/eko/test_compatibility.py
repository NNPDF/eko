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
    new_theory = compatibility.update_theory(theory1)

    assert new_theory["order"][0] == theory1["PTO"] + 1


operator_dict = {"configs": {"ev_op_max_order": 2}}


def test_compatibility_operators():
    _, new_operator = compatibility.update(theory1, operator_dict)

    assert isinstance(new_operator["configs"]["ev_op_max_order"], int)

    with pytest.raises(KeyError):
        compatibility.update(theory1, {})
