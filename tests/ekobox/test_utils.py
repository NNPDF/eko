# -*- coding: utf-8 -*-
import numpy as np
import pytest

import eko
from ekobox import operators_card as oc
from ekobox import theory_card as tc
from ekobox import utils


def test_ekos_product():
    # Generating two ekos
    op1 = oc.generate(
        [60.0, 80.0, 100.0],
        update={
            "interpolation_xgrid": [0.1, 0.5, 1.0],
            "interpolation_polynomial_degree": 1,
        },
    )
    theory1 = tc.generate(0, 5.0)

    op2 = oc.generate(
        [80.0, 100.0, 120.0],
        update={
            "interpolation_xgrid": [0.1, 0.5, 1.0],
            "interpolation_polynomial_degree": 1,
        },
    )
    theory2 = tc.generate(0, 10.0)
    theory_err = tc.generate(0, 5.0)

    eko_ini = eko.run_dglap(theory1, op1)
    eko_fin = eko.run_dglap(theory2, op2)
    # Test_error
    eko_fin_err = eko.run_dglap(theory_err, op2)
    with pytest.raises(ValueError):
        _ = utils.ekos_product(eko_ini, eko_fin_err)
    # product is copied
    eko_res = utils.ekos_product(eko_ini, eko_fin, in_place=False)

    assert eko_res.Q02 == eko_ini.Q02
    np.testing.assert_allclose(eko_res.Q2grid[1:], eko_fin.Q2grid)
    np.testing.assert_allclose(eko_ini[80.0].operator, eko_res[80.0].operator)

    # product overwrites initial
    eko_res2 = utils.ekos_product(eko_ini, eko_fin)

    assert eko_res2.Q02 == eko_ini.Q02
    np.testing.assert_allclose(eko_res2.Q2grid[1:], eko_fin.Q2grid)
    np.testing.assert_allclose(eko_res[80.0].operator, eko_res2[80.0].operator)
