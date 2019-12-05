# -*- coding: utf-8 -*-
# Testing the loading functions
import numpy as np
import pytest

from eko.dglap import run_dglap

# TODO uncomment for now, until they actual work, as they consume time

# TODO define outcome
@pytest.mark.skip(reason="need to define outcome first - too time consuming for now")
def test_loader():
    """Test the loading mechanism"""

    # Allocate a theory from NNPDF database at LO (theory.ID = 71)
    theory = {
        "PTO": 0,
        "FNS": "ZM-VFNS",
        "DAMP": 0,
        "IC": 1,
        "ModEv": "TRN",
        "XIR": 1.0,
        "XIF": 1.0,
        "NfFF": 5,
        "MaxNfAs": 5,
        "MaxNfPdf": 5,
        "Q0": 1.65,
        "alphas": 0.118,
        "Qref": 91.2,
        "QED": 0,
        "alphaqed": 0.007496252,
        "Qedref": 1.777,
        "SxRes": 0,
        "SxOrd": "LL",
        "HQ": "POLE",
        "mc": 1.51,
        "Qmc": 1.51,
        "kcThr": 1.0,
        "mb": 4.92,
        "Qmb": 4.92,
        "kbThr": 1.0,
        "mt": 172.5,
        "Qmt": 172.5,
        "ktThr": 1.0,
        "CKM": "0.97428 0.22530 0.003470 0.22520 0.97345 0.041000 0.00862 0.04030 0.999152",
        "MZ": 91.1876,
        "MW": 80.398,
        "GF": 1.1663787e-05,
        "SIN2TW": 0.23126,
        "TMC": 1,
        "MP": 0.938,
        "Comments": "NNPDF3.1 LO FC",
        "global_nx": 0,
        "EScaleVar": 1,
        "xgrid_size": 7,
        "Q2grid": [1e4],
    }

    # execute DGLAP
    result = run_dglap(theory)

    assert isinstance(result, dict)
    assert "xgrid" in result


@pytest.mark.skip(reason="will fail, so for now skip as it is too time consuming")
def test_loader_benchmark_LHA():
    """benchmark to arXiv:hep-ph/0204316v1"""
    toy_xgrid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9]

    def toy_xuv0(x):
        return 5.107200 * x ** 0.8 * (1.0 - x) ** 3

    xgrid_size = 7
    ret = run_dglap(
        {
            "PTO": 0,
            "alphas": 0.35,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "NfFF": 4,
            "xgrid_size": xgrid_size,
            "targetgrid": toy_xgrid,
            "Q2grid": [1e4],
        }
    )

    # check u_v
    toy_xuv1_xgrid = np.array([toy_xuv0(x) for x in ret["xgrid"]])
    toy_xuv1_grid = np.dot(ret["operators"]["NS"], toy_xuv1_xgrid)
    toy_xuv1_grid_ref = np.array(
        [
            5.7722e-5,
            3.3373e-4,
            1.8724e-3,
            1.0057e-2,
            5.0392e-2,
            2.1955e-1,
            5.7267e-1,
            3.7925e-1,
            1.3476e-1,
            2.3123e-2,
            4.3443e-4,
        ]
    )
    for j in range(xgrid_size):
        assert np.abs(toy_xuv1_grid[j] - toy_xuv1_grid_ref[j]) < 1e-6
