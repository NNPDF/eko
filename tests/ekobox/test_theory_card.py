import pytest

from ekobox import theory_card as tc


def test_generate_theory_card():
    theory = tc.generate(0, 1.0)
    assert theory["PTO"] == 0
    assert theory["Q0"] == 1.0
    assert theory["mt"] == 173.07
    up_err = {"Prova": "Prova"}
    with pytest.raises(ValueError):
        theory = tc.generate(0, 1.0, update=up_err)
    up = {"mb": 132.3, "PTO": 2}
    theory = tc.generate(0, 1.0, update=up)
    assert theory["PTO"] == 2
    assert theory["mb"] == 132.3


def test_dump_load_theory_card(tmp_path, cd):
    with cd(tmp_path):
        theory = tc.generate(2, 12.3, name="debug_theory")
        tc.dump("debug_theory_two", theory)
        theory_loaded = tc.load("debug_theory.yaml")
        theory_two_loaded = tc.load("debug_theory_two.yaml")
        for key in theory.keys():
            assert theory[key] == theory_loaded[key] == theory_two_loaded[key]
