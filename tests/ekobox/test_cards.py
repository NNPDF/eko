import pytest

from ekobox import cards


def test_generate_ocard():
    op = cards.generate_operator([10, 100])
    assert op["Q2grid"] == [10, 100]
    assert op["interpolation_polynomial_degree"] == 4
    up_err = {"Prova": "Prova"}
    with pytest.raises(ValueError):
        op = cards.generate_operator([10], update=up_err)
    up = {"interpolation_polynomial_degree": 2, "interpolation_is_log": False}
    op = cards.generate_operator([100], update=up)
    assert op["Q2grid"] == [100]
    assert op["interpolation_polynomial_degree"] == 2
    assert op["interpolation_is_log"] is False


def test_dump_load_op_card(tmp_path, cd):
    with cd(tmp_path):
        op = cards.generate_operator([100], path=tmp_path / "debug_op.yaml")
        cards.dump(tmp_path / "debug_op_two.yaml", op)
        op_loaded = cards.load("debug_op.yaml")
        op_two_loaded = cards.load("debug_op_two.yaml")
        for key in op.keys():
            assert op[key] == op_loaded[key] == op_two_loaded[key]


def test_generate_theory_card():
    theory = cards.generate_theory(0, 1.0)
    assert theory["PTO"] == 0
    assert theory["Q0"] == 1.0
    assert theory["mt"] == 173.07
    up_err = {"Prova": "Prova"}
    with pytest.raises(ValueError):
        theory = cards.generate_theory(0, 1.0, update=up_err)
    up = {"mb": 132.3, "PTO": 2}
    theory = cards.generate_theory(0, 1.0, update=up)
    assert theory["PTO"] == 2
    assert theory["mb"] == 132.3


def test_dump_load_theory_card(tmp_path, cd):
    with cd(tmp_path):
        theory = cards.generate_theory(2, 12.3, path=tmp_path / "debug_theory.yaml")
        cards.dump(tmp_path / "debug_theory_two.yaml", theory)
        theory_loaded = cards.load("debug_theory.yaml")
        theory_two_loaded = cards.load("debug_theory_two.yaml")
        for key in theory.keys():
            assert theory[key] == theory_loaded[key] == theory_two_loaded[key]
