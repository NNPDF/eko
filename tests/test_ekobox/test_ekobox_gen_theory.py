# -*- coding: utf-8 -*-
import pytest
from utils import cd

from ekobox import gen_theory as g_t


def test_gen_theory_card():
    theory = g_t.gen_theory_card(0, 1.0)
    assert theory["PTO"] == 0
    assert theory["Q0"] == 1.0
    assert theory["mt"] == 173.07
    up_err = {"Prova": "Prova"}
    with pytest.raises(ValueError):
        theory = g_t.gen_theory_card(0, 1.0, update=up_err)
    up = {"mb": 132.3, "PTO": 2}
    theory = g_t.gen_theory_card(0, 1.0, update=up)
    assert theory["PTO"] == 2
    assert theory["mb"] == 132.3


def test_export_load_theory_card(tmp_path):
    with cd(tmp_path):
        theory = g_t.gen_theory_card(2, 12.3, export=True, name="debug_theory")
        g_t.export_theory_card("debug_theory_two", theory)
        theory_loaded = g_t.import_theory_card("debug_theory.yaml")
        theory_two_loaded = g_t.import_theory_card("debug_theory_two.yaml")
        for key in theory.keys():
            assert theory[key] == theory_loaded[key] == theory_two_loaded[key]
