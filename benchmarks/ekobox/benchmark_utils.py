# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ekobox import evol_pdf as ev_p
from ekobox import gen_op as g_o
from ekobox import gen_theory as g_t
from ekobox import utils


@pytest.mark.isolated
def benchmark_ekos_product():
    # Generating two ekos
    op1 = g_o.gen_op_card(
        [60.0, 80.0, 100.0], update={"interpolation_xgrid": [1e-7, 0.01, 0.1, 0.2, 0.3]}
    )
    theory1 = g_t.gen_theory_card(0, 5.0)

    op2 = g_o.gen_op_card(
        [80.0, 100.0, 120.0],
        update={"interpolation_xgrid": [1e-7, 0.01, 0.1, 0.2, 0.3]},
    )
    theory2 = g_t.gen_theory_card(0, 10.0)
    theory_err = g_t.gen_theory_card(0, 5.0)

    eko_ini = ev_p.gen_out(theory1, op1)
    eko_fin = ev_p.gen_out(theory2, op2)
    # Test_error
    eko_fin_err = ev_p.gen_out(theory_err, op2)
    with pytest.raises(ValueError):
        _ = utils.ekos_product(eko_ini, eko_fin_err)
    # product is copied
    eko_res = utils.ekos_product(eko_ini, eko_fin, in_place=False)
    # product overwrites initial
    eko_res2 = utils.ekos_product(eko_ini, eko_fin)
    np.testing.assert_allclose(
        eko_res["Q2grid"][80.0]["operators"], eko_res2["Q2grid"][80.0]["operators"]
    )
    np.testing.assert_allclose(eko_res2["q2_ref"], eko_ini["q2_ref"])
    np.testing.assert_allclose(
        list(eko_res2["Q2grid"].keys()), list(eko_fin["Q2grid"].keys())
    )

    np.testing.assert_allclose(
        eko_ini["Q2grid"][80.0]["operators"], eko_res2["Q2grid"][80.0]["operators"]
    )
    np.testing.assert_allclose(eko_res["q2_ref"], eko_ini["q2_ref"])
    np.testing.assert_allclose(
        list(eko_res["Q2grid"].keys()), list(eko_fin["Q2grid"].keys())
    )
