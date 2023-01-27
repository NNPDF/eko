"""The unpolarized time-like next-to-next-to-leading-order (NNLO) Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np
from eko import constants

@nb.njit(cache=True)
def gamma_nsp():
    """Computes the NNLO non-singlet positive anomalous dimension.
    
    """
    return

@nb.njit(cache=True)
def gamma_nsm():
    """Computes the NNLO non-singlet negative anomalous dimension.
    
    """
    return

@nb.njit(cache=True)
def gamma_nsv():
    """Computes the NNLO non-singlet valence anomalous dimension.
    
    """

@nb.njit(cache=True)
def gamma_qqs():
    """Computes the NNLO single quark-quark anomalous dimension.
    
    """
    return

@nb.njit(cache=True)
def gamma_qg():
    """Computes the NNLO quark-gluon anomalous dimension.
    
    """
    return

@nb.njit(cache=True)
def gamma_gq():
    """Computes the NNLO gluon-quark anomalous dimension.
    
    """
    return

@nb.njit(cache=True)
def gamma_gg():
    """Computes the NNLO gluon-gluon anomalous dimension.
    
    """
    return

@nb.njit(cache=True)
def gamma_singlet():
    """Computes the NNLO singlet anomalous dimension matrix.
    
    """
    return