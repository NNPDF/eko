# -*- coding: utf-8 -*-
# Testing the loading functions
from eko.dglap import run_dglap

def test_loader():
    """Test the loading mechanism"""

    # Allocate a theory from NNPDF database at LO (theory.ID = 71)
    theory = {
        'PTO': 0,
        'FNS': 'ZM-VFNS',
        'DAMP': 0,
        'IC': 1,
        'ModEv': 'TRN',
        'XIR': 1.0,
        'XIF': 1.0,
        'NfFF': 5,
        'MaxNfAs': 5,
        'MaxNfPdf': 5,
        'Q0': 1.65,
        'alphas': 0.118,
        'Qref': 91.2,
        'QED': 0,
        'alphaqed': 0.007496252,
        'Qedref': 1.777,
        'SxRes': 0,
        'SxOrd': 'LL',
        'HQ': 'POLE',
        'mc': 1.51,
        'Qmc': 1.51,
        'kcThr': 1.0,
        'mb': 4.92,
        'Qmb': 4.92,
        'kbThr': 1.0,
        'mt': 172.5,
        'Qmt': 172.5,
        'ktThr': 1.0,
        'CKM': '0.97428 0.22530 0.003470 0.22520 0.97345 0.041000 0.00862 0.04030 0.999152',
        'MZ': 91.1876,
        'MW': 80.398,
        'GF': 1.1663787e-05,
        'SIN2TW': 0.23126,
        'TMC': 1,
        'MP': 0.938,
        'Comments': 'NNPDF3.1 LO FC',
        'global_nx': 0,
        'EScaleVar': 1
        }

    # esecute DGLAP
    result = run_dglap(theory)

    assert result == 0
