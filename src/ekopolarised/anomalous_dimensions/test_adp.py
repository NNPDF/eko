
import numba as nb
import numpy as np

from ekopolarised import basis_rotation as br
from ekopolarised import harmonics
from ekopolarised import anomalous_dimensions as ad
from ekopolarised.anomalous_dimensions.__init__ import gamma_singlet
import warnings
from numpy.testing import assert_allclose, assert_almost_equal, assert_raises




def test_gamma_s():
    nf = 3
    p = True
    # LO
    first_order= gamma_singlet((1, 0), br.non_singlet_pids_map["ns+"], nf, p)
    print(first_order) 

test_gamma_s()