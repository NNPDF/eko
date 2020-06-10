# -*- coding: utf-8 -*-
"""
    Checks that the operator grid works as intended
    These test can be slow as they require the computation of several values of Q
    But they should be fast as the grid is very small.
    It does *not* test whether the result is correct, it can just test that it is sane
"""

import pytest
import numpy as np
import eko.interpolation as interpolation
from eko.strong_coupling import StrongCoupling
from eko.constants import Constants
from eko.thresholds import Threshold
from eko.kernel_generation import KernelDispatcher
from eko.operator_grid import OperatorGrid

N_LOW = 3
N_MID = 3


def generate_fake_pdf():
    basis = ["V", "V3", "V8", "V15", "T3", "T15", "S", "g"]
    len_grid = N_LOW + N_MID - 1
    pdf_m = {}
    for i in basis:
        pdf_m[i] = np.random.rand(len_grid)
        pdf_m[i].sort()
    pdf = {"metadata": "evolbasis", "members": pdf_m}
    return pdf


def get_setup():
    n_low = N_LOW
    n_mid = N_MID
    xgrid_low = np.geomspace(1e-7, 1.0 if n_mid == 0 else 0.1, n_low)
    xgrid_mid = np.linspace(0.1, 1.0, n_mid)
    xgrid_high = np.array([])
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid, xgrid_high)))

    setup = {
        "alphas": 0.35,
        "xgrid": xgrid,
        "polynom_rank": 4,
    }
    return setup


def generate_fake_grid(q2_ref=2.0, q2alpha=None, thresholds=None):
    if thresholds is None:
        scheme = "FFNS"
        nf = 5
    else:
        scheme = "VFNS"
        nf = None
    if q2alpha is None:
        q2alpha = q2_ref
    setup = get_setup()
    constants = Constants()
    xgrid = setup["xgrid"]
    polynom_rank = setup["polynom_rank"]
    basis_function_dispatcher = interpolation.InterpolatorDispatcher(
        xgrid, polynom_rank, log=True
    )
    kernel_dispatcher = KernelDispatcher(basis_function_dispatcher, constants)
    threshold_holder = Threshold(
        q2_ref=q2_ref, scheme=scheme, threshold_list=thresholds, nf=nf
    )

    # Now generate the a_s class
    alpha_ref = setup["alphas"]
    a_s = StrongCoupling(constants, alpha_ref, q2alpha, threshold_holder)
    opgrid = OperatorGrid(threshold_holder, a_s, kernel_dispatcher, xgrid)
    return opgrid


def test_sanity():
    """ Sanity checks for the input"""
    thresholds = [4, 15]
    opgrid = generate_fake_grid(thresholds=thresholds)
    # Check that an operator grid with the correct number of regions was created
    nregs = len(opgrid._op_masters)  # pylint: disable=protected-access
    assert nregs == len(thresholds) + 1
    # Check that the errors work
    with pytest.raises(ValueError):
        opgrid.set_q2_limits(-1, 4)
    with pytest.raises(ValueError):
        opgrid.set_q2_limits(-1, -4)
    with pytest.raises(ValueError):
        opgrid.set_q2_limits(4, 1)
    with pytest.raises(ValueError):
        bad_grid = [100, -6, 3]
        _ = opgrid.compute_q2grid(bad_grid)


def test_grid_computation_VFNS():
    """ Checks that the grid can be computed """
    thresholds = [4, 15]
    opgrid = generate_fake_grid(thresholds=thresholds)
    qgrid_check = [0.3, 5]
    operators = opgrid.compute_q2grid(qgrid_check)
    assert len(operators) == len(qgrid_check)
    # Check that the operators can act on pdfs
    pdf = generate_fake_pdf()
    _return_1 = operators[0](pdf)
    _return_2 = operators[1](pdf)
