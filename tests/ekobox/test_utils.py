import copy

import numpy as np
import pytest

import eko
from eko import EKO, interpolation
from ekobox import cards, utils


def test_ekos_product(tmp_path):
    # Generating two ekos
    mu01 = 5.0
    mu2grid1 = np.array([60.0, 80.0, 100.0])
    xgrid = interpolation.XGrid([0.1, 0.5, 1.0])

    theory = cards.example.theory()
    theory.order = (1, 0)

    op1 = cards.example.operator()
    op1.mu0 = mu01
    op1.mu2grid = mu2grid1
    op1.rotations.xgrid = xgrid
    op1.configs.interpolation_polynomial_degree = 1

    mu02 = 10.0
    mu2grid2 = np.array([80.0, 100.0, 120.0])

    op2 = cards.example.operator()
    op2.mu0 = mu02
    op2.mu2grid = mu2grid2
    op2.rotations.xgrid = xgrid
    op2.configs.interpolation_polynomial_degree = 1

    op_err = copy.deepcopy(op2)
    op_err.mu0 = mu01

    ini_path = tmp_path / "ini.tar"
    eko.solve(theory, op1, path=ini_path)
    fin_path = tmp_path / "fin.tar"
    eko.solve(theory, op2, path=fin_path)
    # Test_error
    fin_err_path = tmp_path / "fin_err.tar"
    eko_fin_err = eko.solve(theory, op_err, path=fin_err_path)
    with EKO.edit(ini_path) as eko_ini:
        with EKO.edit(fin_path) as eko_fin:
            with EKO.read(fin_err_path) as eko_fin_err:
                with pytest.raises(ValueError):
                    _ = utils.ekos_product(eko_ini, eko_fin_err)
                # product is copied
                res_path = tmp_path / "eko_res.tar"
                eko_fin[120.0].error = None  # drop one set of errors
                utils.ekos_product(eko_ini, eko_fin, path=res_path)

                with EKO.read(res_path) as eko_res:
                    assert eko_res.operator_card.mu20 == eko_ini.operator_card.mu20
                    np.testing.assert_allclose(eko_res.mu2grid[1:], eko_fin.mu2grid)
                    np.testing.assert_allclose(
                        eko_ini[80.0].operator, eko_res[80.0].operator
                    )

                    # product overwrites initial
                    utils.ekos_product(eko_ini, eko_fin)

                    np.testing.assert_allclose(eko_ini.mu2grid[1:], eko_fin.mu2grid)
                    np.testing.assert_allclose(
                        eko_res[80.0].operator, eko_ini[80.0].operator
                    )
                    utils.ekos_product(eko_ini, eko_fin)

                    assert eko_res[120.0].error is None
