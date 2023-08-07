import copy

import numpy as np
import pytest

import eko
from eko import EKO, interpolation
from ekobox import cards, utils


def test_ekos_product(tmp_path):
    # Generating two ekos
    mu01 = 5.0
    mugrid1 = [(1.0, 3), (8.0, 5), (10.0, 5)]
    xgrid = interpolation.XGrid([0.1, 0.5, 1.0])

    theory = cards.example.theory()
    theory.order = (1, 0)
    theory.heavy.num_flavs_init = 5

    op1 = cards.example.operator()
    op1.mu0 = mu01
    op1.mugrid = mugrid1
    op1.xgrid = xgrid
    op1.configs.interpolation_polynomial_degree = 1

    mu0 = 10.0
    mugrid2 = [(8.0, 5), (10.0, 5), (12.0, 5)]

    op2 = cards.example.operator()
    op2.mu0 = mu0
    op2.mugrid = mugrid2
    op2.xgrid = xgrid
    op2.configs.interpolation_polynomial_degree = 1

    op_err = copy.deepcopy(op2)
    op_err.mu0 = mu01

    mu2first = (mugrid2[0][0] ** 2, mugrid2[0][1])

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
                utils.ekos_product(eko_ini, eko_fin, path=res_path)

                with EKO.read(res_path) as eko_res:
                    assert eko_res.operator_card.mu20 == eko_ini.operator_card.mu20
                    np.testing.assert_allclose(
                        sorted(eko_res.mu2grid)[1:], sorted(eko_fin.mu2grid)
                    )
                    np.testing.assert_allclose(
                        eko_ini[mu2first].operator, eko_res[mu2first].operator
                    )

                    # product overwrites initial
                    utils.ekos_product(eko_ini, eko_fin)

                    np.testing.assert_allclose(
                        sorted(eko_ini.mu2grid)[1:], sorted(eko_fin.mu2grid)
                    )
                    np.testing.assert_allclose(
                        eko_res[mu2first].operator, eko_ini[mu2first].operator
                    )
                    utils.ekos_product(eko_ini, eko_fin)
