import math

import pytest

from ekobox import cards


def test_generate_ocard():
    mu0 = 1.65
    mu2grid = [10.0, 100.0]
    op = cards.example.operator()
    op.mu0 = mu0
    op.mu2grid = mu2grid
    assert pytest.approx(op.mu2grid) == mu2grid
    assert op.configs.interpolation_polynomial_degree == 4
    mu2grid1 = [100.0]
    op = cards.example.operator()
    op.mu0 = mu0
    op.mu2grid = mu2grid1
    op.configs.interpolation_polynomial_degree = 2
    op.configs.interpolation_is_log = False
    assert op.mu2grid == mu2grid1
    assert op.configs.interpolation_polynomial_degree == 2
    assert op.configs.interpolation_is_log is False


def test_dump_load_op_card(tmp_path, cd):
    mu0 = 1.65
    with cd(tmp_path):
        path1 = tmp_path / "debug_op.yaml"
        path2 = tmp_path / "debug_op_two.yaml"
        op = cards.example.operator()
        op.mu0 = mu0
        cards.dump(op.raw, path1)
        cards.dump(op.raw, path2)
        op_loaded = cards.load(path1)
        op_two_loaded = cards.load(path2)
        for key, value in op.raw.items():
            assert value == op_loaded[key] == op_two_loaded[key]


def test_generate_theory_card():
    theory = cards.example.theory()
    assert hasattr(theory, "order")
    assert theory.quark_masses.t.value == 173.07
    theory.order = (2, 0)
    assert theory.order[0] == 2


def containsnan(obj) -> bool:
    if isinstance(obj, dict):
        return containsnan(list(obj.values()))
    if isinstance(obj, list):
        return any(containsnan(el) for el in obj)
    if not isinstance(obj, float):
        return False

    return math.isnan(obj)


def test_dump_load_theory_card(tmp_path, cd):
    with cd(tmp_path):
        theory = cards.example.theory()
        theory.order = (3, 0)
        cards.dump(theory.raw, path=tmp_path / "debug_theory.yaml")
        cards.dump(theory.raw, tmp_path / "debug_theory_two.yaml")
        theory_loaded = cards.load("debug_theory.yaml")
        theory_two_loaded = cards.load("debug_theory_two.yaml")
        for key, value in theory.raw.items():
            if containsnan(value):
                continue
            assert value == theory_loaded[key] == theory_two_loaded[key]
