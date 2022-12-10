import numpy as np
import pytest

import eko
from eko.io import runcards
from ekobox import cards, utils


def test_ekos_product(tmp_path):
    # Generating two ekos
    op1 = cards.generate_operator(
        [60.0, 80.0, 100.0],
        update={
            "interpolation_xgrid": [0.1, 0.5, 1.0],
            "interpolation_polynomial_degree": 1,
        },
    )
    theory1 = cards.generate_theory(0, 5.0)

    op2 = cards.generate_operator(
        [80.0, 100.0, 120.0],
        update={
            "interpolation_xgrid": [0.1, 0.5, 1.0],
            "interpolation_polynomial_degree": 1,
        },
    )
    theory2 = cards.generate_theory(0, 10.0)
    theory_err = cards.generate_theory(0, 5.0)

    new_theory1, new_op1 = runcards.update(theory1, op1)
    new_theory2, new_op2 = runcards.update(theory2, op2)

    eko_ini = eko.solve(new_theory1, new_op1, path=tmp_path / "ini.tar")
    eko_fin = eko.solve(new_theory2, new_op2, path=tmp_path / "fin.tar")
    # Test_error
    eko_fin_err = eko.solve(theory_err, op2, path=tmp_path / "fin_err.tar")
    with pytest.raises(ValueError):
        _ = utils.ekos_product(eko_ini, eko_fin_err)
    # product is copied
    eko_res = utils.ekos_product(eko_ini, eko_fin, in_place=False)

    assert eko_res.operator_card.mu20 == eko_ini.operator_card.mu20
    np.testing.assert_allclose(eko_res.mu2grid[1:], eko_fin.mu2grid)
    np.testing.assert_allclose(eko_ini[80.0].operator, eko_res[80.0].operator)

    # product overwrites initial
    eko_res2 = utils.ekos_product(eko_ini, eko_fin)

    assert eko_res2.operator_card.mu20 == eko_ini.operator_card.mu20
    np.testing.assert_allclose(eko_res2.mu2grid[1:], eko_fin.mu2grid)
    np.testing.assert_allclose(eko_res[80.0].operator, eko_res2[80.0].operator)
