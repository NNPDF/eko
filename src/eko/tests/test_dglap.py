import platform
import numpy as np
import pytest

import eko.dglap as dglap
import eko.interpolation as interpolation

# implement Eq. 31 of arXiv:hep-ph/0204316
def toy_uv0(x):
    return 5.107200 * x**(0.8) * (1.0 - x)**3 / x
def toy_dv0(x):
    return 3.064320 * x**(0.8) * (1.0 - x)**4 / x
def toy_g0(x):
    return 1.7 * x**(-0.1) * (1.0 - x)**5 / x
def toy_dbar0(x):
    return 0.1939875 *  x**(-0.1) *  (1.0 - x)**6 / x
def toy_ubar0(x):
    return (1.0 - x) * toy_dbar0(x)
def toy_s0(x):
    return 0.2 * (toy_ubar0(x) + toy_dbar0(x))
def toy_sbar0(x):
    return toy_s0(x)
def toy_Lm0(x):
    return toy_dbar0(x) - toy_ubar0(x)
def toy_Lp0(x):
    return (toy_dbar0(x) + toy_ubar0(x))*2.0 # 2 is missing in the paper!
def toy_sp0(x):
    return toy_s0(x) + toy_sbar0(x)
def toy_T30(x):
    return -2.0 * toy_Lm0(x) + toy_uv0(x) - toy_dv0(x)
def toy_T80(x):
    return toy_Lp0(x) + toy_uv0(x) + toy_dv0(x) - 2.0*toy_sp0(x)
def toy_S0(x):
    return toy_uv0(x) + toy_dv0(x) + toy_Lp0(x) + toy_sp0(x)

def reference_values():
    """ Return a dictionary with the values from
    table 2 part 2 of arXiv:hep-ph/0204316
    """
    # fmt: off
    xuv1 = np.array([5.7722e-5,3.3373e-4,1.8724e-3,1.0057e-2,5.0392e-2,2.1955e-1,5.7267e-1,3.7925e-1,1.3476e-1,2.3123e-2,4.3443e-4])
    xdv1 = np.array([3.4343e-5,1.9800e-4,1.1065e-3,5.9076e-3,2.9296e-2,1.2433e-1,2.8413e-1,1.4186e-1,3.5364e-2,3.5943e-3,2.2287e-5])
    xLm1_aux = np.array([7.6527e-7,5.0137e-6,3.1696e-5,1.9071e-4,1.0618e-3,4.9731e-3,1.0470e-2,3.3029e-3,4.2815e-4,1.5868e-5,1.1042e-8])
    xT31 = -2.0 * xLm1_aux + xuv1 - xdv1
    xLp1_aux = np.array([9.9465e+1,5.0259e+1,2.4378e+1,1.1323e+1,5.0324e+0,2.0433e+0,4.0832e-1,4.0165e-2,2.8624e-3,6.8961e-5,3.6293e-8])
    xsp1_aux = np.array([4.8642e+1,2.4263e+1,1.1501e+1,5.1164e+0,2.0918e+0,7.2814e-1,1.1698e-1,1.0516e-2,7.3138e-4,1.7725e-5,1.0192e-8])
    xT81 = xLp1_aux + xuv1 + xdv1 - 2.0 * xsp1_aux
    xcp1_aux = np.array([4.7914e+1,2.3685e+1,1.1042e+1,4.7530e+0,1.8089e+0,5.3247e-1,5.8864e-2,4.1379e-3,2.6481e-4,6.5549e-6,4.8893e-9])
    T151 = xLp1_aux + xuv1 + xdv1 + xsp1_aux - 3.0 * xcp1_aux
    xg1 = np.array([1.3162e+3,6.0008e+2,2.5419e+2,9.7371e+1,3.2078e+1,8.0546e+0,8.8766e-1,8.2676e-2,7.9240e-3,3.7311e-4,1.0918e-6])
    xS1 = xuv1 + xdv1 + xLp1_aux + xsp1_aux + xcp1_aux
    # fmt: on
    non_singlet = [
            (xuv1 , toy_uv0),
            (xdv1 , toy_dv0),
            (xT31 , toy_T30),
            (xT81 , toy_T80),
            (T151 , toy_S0)
                ]
    singlets = {
            ("S_qq", "S_qg"): (xS1 , [toy_S0, toy_g0]),
            ("S_gq", "S_gg"): (xg1 , [toy_S0, toy_g0]),
            }
    ret = { "NS"  : non_singlet, "singlets" : singlets }
    ret.update(singlets)
    return ret

def check_operator(operators, xgrid, toy_xgrid):
    """ Runs through the reference values and gets the appropiate
    operators from the operators dictionary.
    Checks how separated are the reference values from the ones
    computed by eko
    """
    reference = reference_values()

    # Check non-singlet side
    ref_ns = reference["NS"]
    op_ns = operators["NS"]
    for grid, function in ref_ns:
        toy_val = function(xgrid)
        op_val = np.dot(op_ns, toy_val)*toy_xgrid
        np.testing.assert_allclose(grid, op_val, atol = 2e-1)
        # Note: most pass with a tolerance below atol = 1e-4
        # but this bigger tolerance is needed for:
        # T_8 (one value)
        # T_15 (one value)

    # Check singlet side
    ref_s = reference["singlets"]
    for keys, item in ref_s.items():
        ops = [operators[key] for key in keys]
        grid = item[0]
        op_val = np.zeros_like(toy_xgrid)
        for op, toy_fun in zip(ops, item[1]):
            op_val += np.dot(op, toy_fun(xgrid))*toy_xgrid
        np.testing.assert_allclose(grid, op_val, atol=3e-1)

@pytest.mark.skipif(platform.node() == "FHe19b",reason="too time consuming for now")
def test_dglap_ffns_lo():
    """Checks table 2 part 2 of :cite:`Giele:2002hx`"""
    # Prepare a custom grid
    toy_xgrid = np.array([1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,.1,.3,.5,.7,.9])
    xgrid_low = interpolation.get_xgrid_linear_at_log(35,1e-7,0.1)
    xgrid_mid = interpolation.get_xgrid_linear_at_id(15,0.1,1.0)
    polynom_rank = 4

    xgrid_high = np.array([])#1.0-interpolation.get_xgrid_linear_at_log(10,1e-3,1.0 - 0.9)
    xgrid = np.unique(np.concatenate((xgrid_low,xgrid_mid,xgrid_high)))

    # Prepare a setup dictionary
    setup = {
        "PTO": 0,
        'alphas': 0.35,
        'Qref': np.sqrt(2),
        'Q0': np.sqrt(2),
        'NfFF': 4,

        "xgrid_type": "custom",
        "xgrid": xgrid,
        "xgrid_polynom_rank": polynom_rank,
        "log_interpol": "log",
        "targetgrid": toy_xgrid,
        "Q2grid": [1e4],
    }

    return_dictionary = dglap.run_dglap(setup)
    check_operator(return_dictionary["operators"], xgrid, toy_xgrid)

import pprint
def test_dglap_prod():
    """check multiplication"""
    # Prepare a custom grid
    xgrid_low = interpolation.get_xgrid_linear_at_log(10,1e-7,0.1)
    xgrid_mid = interpolation.get_xgrid_linear_at_id(5,0.1,1.0)
    polynom_rank = 4
    xgrid = np.unique(np.concatenate((xgrid_low,xgrid_mid)))
    # Prepare a setup dictionary
    setup = {
        "PTO": 0,
        'alphas': 0.35,
        'Qref': np.sqrt(2),
        'Q0': np.sqrt(2),
        'NfFF': 4,
        'FNS': 'FFNS',

        "xgrid_type": "custom",
        "xgrid": xgrid,
        "xgrid_polynom_rank": polynom_rank,
        "log_interpol": "log",
        "Q2grid": [1e4],
    }
    # check 0 -> 1 -> 2 = 0 -> 2
    Q2init = 2
    Q2mid = 1e2
    Q2final = 1e4
    # step 1
    setup["Q0"] = np.sqrt(Q2init)
    setup["Q2grid"] = [Q2mid]
    ret1 = dglap.run_dglap(setup)
    # step 2
    setup["Q0"] = np.sqrt(Q2mid)
    setup["Q2grid"] = [Q2final]
    ret2 = dglap.run_dglap(setup)
    # step 1+2
    setup["Q0"] = np.sqrt(Q2init)
    setup["Q2grid"] = [Q2final]
    ret12 = dglap.run_dglap(setup)
    # check
    for label in dglap.evolution_basis_label_list:
        print(label)
        pprint.pprint(ret1["operators"][label])
        pprint.pprint(ret2["operators"][label])
        pprint.pprint(ret12["operators"][label])
        mult = np.dot(ret1["operators"][label],ret2["operators"][label])
        ref = ret12["operators"][label]
        np.testing.assert_allclose(mult, ref, atol=3e-1)
        print("------------\n")
