from eko.runner.compute import compute
from eko.runner.recipes import Recipe


def test_compute():
    recipe = Recipe("ciao")
    part = compute(recipe)

    assert hasattr(part, "operator")
