# -*- coding: utf-8 -*-
import pytest
from utils import cd

from ekobox import gen_op as g_o


def test_gen_op_card():
    op = g_o.gen_op_card([10, 100])
    assert op["Q2grid"] == [10, 100]
    assert op["interpolation_polynomial_degree"] == 4
    up_err = {"Prova": "Prova"}
    with pytest.raises(ValueError):
        op = g_o.gen_op_card([10], update=up_err)
    up = {"interpolation_polynomial_degree": 2, "interpolation_is_log": False}
    op = g_o.gen_op_card([100], update=up)
    assert op["Q2grid"] == [100]
    assert op["interpolation_polynomial_degree"] == 2
    assert op["interpolation_is_log"] == False


def test_export_load_op_card(tmp_path):
    with cd(tmp_path):
        op = g_o.gen_op_card([100], name="debug_op")
        g_o.export_op_card("debug_op_two", op)
        op_loaded = g_o.import_op_card("debug_op.yaml")
        op_two_loaded = g_o.import_op_card("debug_op_two.yaml")
        for key in op.keys():
            assert op[key] == op_loaded[key] == op_two_loaded[key]
