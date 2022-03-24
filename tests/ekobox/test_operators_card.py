# -*- coding: utf-8 -*-
import pytest

from ekobox import operators_card as oc


def test_generate_ocard():
    op = oc.generate([10, 100])
    assert op["Q2grid"] == [10, 100]
    assert op["configs"]["interpolation_polynomial_degree"] == 4
    up_err = {"Prova": "Prova"}
    with pytest.raises(ValueError):
        op = oc.generate([10], update=up_err)
    up = {
        "configs": {"interpolation_polynomial_degree": 2, "interpolation_is_log": False}
    }
    op = oc.generate([100], update=up)
    assert op["Q2grid"] == [100]
    assert op["configs"]["interpolation_polynomial_degree"] == 2
    assert op["configs"]["interpolation_is_log"] is False


def test_dump_load_op_card(tmp_path, cd):
    with cd(tmp_path):
        op = oc.generate([100], name="debug_op")
        oc.dump("debug_op_two", op)
        op_loaded = oc.load("debug_op.yaml")
        op_two_loaded = oc.load("debug_op_two.yaml")
        for key in op.keys():
            assert op[key] == op_loaded[key] == op_two_loaded[key]
