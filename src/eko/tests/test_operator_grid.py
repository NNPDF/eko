"""
    Checks that the operator grid works as intended
    These test can be slow as they require the computation of several values of Q
    But they should be fast as the grid is very small.
    It does *not* test whether the result is correct, it can just test that it is sane
"""

# TODO: regression tests
import pytest
import numpy as np
import eko.interpolation as interpolation
from eko.alpha_s import StrongCoupling
from eko.constants import Constants
from eko.thresholds import Threshold
from eko.kernel_generation import KernelDispatcher
from eko.operator_grid import OperatorGrid        
                                                  
N_LOW = 3          
N_MID = 3
                                                  
def generate_fake_pdf():                           
    basis = ['V', 'V3', 'V8', 'V15', 'T3', 'T15', 'S', 'g']
    len_grid = N_LOW+N_MID-1
    pdf_m = {}
    for i in basis:
        pdf_m[i] = np.random.rand(len_grid)
        pdf_m[i].sort()
    pdf = {'metadata' : 'evolbasis', 'members': pdf_m}
    return pdf

def get_setup():
    n_low = N_LOW
    n_mid = N_MID
    xgrid_low = interpolation.get_xgrid_linear_at_log(
            n_low, 1e-7, 1.0 if n_mid == 0 else 0.1
        )
    xgrid_mid = interpolation.get_xgrid_linear_at_id(n_mid, 0.1, 1.0)
    xgrid_high = np.array([])
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid, xgrid_high)))

    setup = {
            "alphas": 0.35,
            "xgrid": xgrid,
            "polynom_rank": 4,
            }
    return setup

def generate_fake_grid(qref_sq = 2.0, qalpha = None, thresholds = None):
    if thresholds is None:
        scheme = 'FFNS'
        nf = 5
    else:
        scheme = 'VFNS'
        nf = None
    if qalpha is None:
        qalpha = qref_sq
    setup = get_setup()
    constants = Constants()
    xgrid = setup["xgrid"]
    polynom_rank = setup["polynom_rank"]
    basis_function_dispatcher = interpolation.InterpolatorDispatcher(
            xgrid, polynom_rank, log=True
    )
    kernel_dispatcher = KernelDispatcher(basis_function_dispatcher, constants)
    threshold_holder = Threshold(qref = qref_sq, scheme = scheme, threshold_list=thresholds, nf=nf)

    # Now generate the operator alpha_s class
    alpha_ref = setup['alphas']
    alpha_s = StrongCoupling(constants, alpha_ref, qalpha, threshold_holder)
    opgrid = OperatorGrid(threshold_holder, alpha_s, kernel_dispatcher, xgrid)
    return opgrid

def test_sanity():
    """ Sanity checks for the input"""
    thresholds = [4, 15]
    opgrid = generate_fake_grid(thresholds=thresholds)
    # Check that an operator grid with the correct number of regions was created
    nregs = len(opgrid._op_masters)
    assert nregs == len(thresholds)+1
    # Check that the errors work
    with pytest.raises(ValueError) as vexp:
        opgrid.set_q_limits(-1, 4)
    with pytest.raises(ValueError) as vexp:
        opgrid.set_q_limits(-1, -4)
    with pytest.raises(ValueError) as vexp:
        opgrid.set_q_limits(4, 1)
    with pytest.raises(ValueError) as vexp:
        bad_grid = [100, -6, 3]
        _ = opgrid.compute_qgrid(bad_grid)

def test_grid_computation_VFNS():
    """ Checks that the grid can be computed """
    thresholds = [4, 15]
    opgrid = generate_fake_grid(thresholds=thresholds)
    qgrid_check = [0.3, 5]
    operators = opgrid.compute_qgrid(qgrid_check)
    assert len(operators) == len(qgrid_check)
    # Check that the operators can act on pdfs
    pdf = generate_fake_pdf()
    return_1 = operators[0](pdf)
    return_2 = operators[1](pdf)

def test_grid_computation_FFNS():
    """ Check that the results from the grid are consistent """
    pdf = generate_fake_pdf()
    ref_1 = 2
    ref_2 = 50
    opgrid_1 = generate_fake_grid(qref_sq=ref_1, qalpha=ref_1)
    qgrid_1 = [ref_2,100]
    results_1 = opgrid_1.compute_qgrid(qgrid_1)
# TODO: for this test we have to use some more reasonable input
#     opgrid_2 = generate_fake_grid(qref_sq=ref_2, qalpha = ref_1)
#     qgrid_2 = [100]
#     results_2 = opgrid_2.compute_qgrid(qgrid_2)
#     pdf_path_1 = results_1[1](pdf)['members']
#     pdf_path_2 = results_2[0](results_1[0](pdf))['members']
            

if __name__ == "__main__":
    test_grid_computation_FFNS()
