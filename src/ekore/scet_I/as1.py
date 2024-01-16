r"""SCET 1 kernel entries.

"""
import numba as nb
import numpy as np

from eko.constants import CF
from ..harmonics import cache as c

@nb.njit(cache=True)
def A_gg(n, order, cache):
    r"""

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    L : int
        logarithmic order

    Returns
    -------
    complex
        |NLO| heavy-heavy |OME| :math:`A_{HH}^{(1)}`

    """

    S1n = c.get(c.S1, cache, n)
    S1nm1 = c.get(c.S1, cache, n-1)
    S1np1 = c.get(c.S1, cache, n+1)
    S1np2 = c.get(c.S1, cache, n+2)

    S2nm1 = c.get(c.S2, cache, n-1)
    S2n = c.get(c.S2, cache, n)
    S2np1 = c.get(c.S2, cache, n+1)
    S2nm1 = c.get(c.S2, cache, n-1)
    S2np2 = c.get(c.S2, cache, n+2)
    S2nm2 = c.get(c.S2, cache, n-2)

    S11nm1 = S1nm1 + 1./n

    if order==(1,0):

        res = 3. * ( - (S1nm1 / (-1 + n) ) + ( 2. * S1n ) / n - S1np1 / (1 + n) + 
        S1np2 / (2 + n) ) 
        - 3. * ( -1./3. * np.pi**2 + S2nm2 - 2. * ( -1./6. * np.pi**2 + S2nm1 ) 
        + 3. * ( -1/6*np.pi**2 + S2n ) - 2. * ( -1./6. * np.pi**2 + S2np1 ) 
        + S2np2) + 3. * S11nm1
        #HSum[{1, 1}, -1 + n]

    if order==(1,1):

        res = - 3. * ( 3. * ( (-1 + n)**(-1) - n**(-1) ) - 3./n +  3. / (1 + n) - 
        (2. + n)**(-1) - 2. * ((-1. + n)**(-1) - 2./n + (1. + n)**(-1))) + 3. * S1nm1
 
    if order==(1,2):

        res = 3./2. 
    
    return res


@nb.njit(cache=True)
def A_gq(n, order, cache):
    r"""
    Parameters
    ----------
    n : complex
        Mellin moment
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| gluon-heavy |OME| :math:`A_{gH}^{(1)}`

    """

    S1n = c.get(c.S1, cache, n)
    S1nm1 = c.get(c.S1, cache, n-1)
    S1np1 = c.get(c.S1, cache, n+1)
    
    if order==(1,0):

        2. / ( 3. * (1 + n) ) - ( 2. * (-2. / (-1. + n)**2 + 2. / n**2 - (1. + n)**(-2) ) ) / 3. 
        + ( 2 * ( ( -2. * S1nm1 ) / (-1. + n) + (2. * S1n)/n 
        - S1np1/(1. + n)) ) / 3.


    if order==(1,1):

        res = ( -2. * ( 2./(-1. + n) - 2./n + (1. + n)**(-1) ) ) / 3.
 
    if order==(1,2):
        
        res = 0. 

    return res


@nb.njit(cache=True)
def A_qg(n, order, cache):
    r"""|

    Parameters
    ----------
    n : complex
        Mellin moment
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| heavy-gluon |OME| :math:`A_{Hg}^{S,(1)}`

    """

    S1n = c.get(c.S1, cache, n)
    S1nm1 = c.get(c.S1, cache, n-1)
    S1np1 = c.get(c.S1, cache, n+1)
    S2nm1 = c.get(c.S2, cache, n-1)
    S1np2 = c.get(c.S1, cache, n+2)

    
    if order==(1,0):

        ( n**(-2) - 2. / (1. + n)**2 + 2. / (2. + n)**2)/4. 
        + ( (1. + n)**(-1) - (2. + n)**(-1)) / 2.  
        + (- ( S1n / n ) + (2. * S1np1) / (1. + n) 
        - (2 * S1np2) / (2. + n) )/4.



    if order==(1,1):

        res = (-n**(-1) + 2. * (n**(-1) - (1. + n)**(-1)) 
        - 2. * (n**(-1) - 2./(1. + n) 
        + (2. + n)**(-1)))/4.

 
    if order==(1,2):
        
        res = 0. 

    return res


@nb.njit(cache=True)
def A_qq(n, order, cache):
    r"""

    Parameters
    ----------
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        |NLO| gluon-gluon |OME| :math:`A_{gg,H}^{S,(1)}`

    """

    S1n = c.get(c.S1, cache, n)
    S1nm1 = c.get(c.S1, cache, n-1)
    S1np1 = c.get(c.S1, cache, n+1)
    S1np2 = c.get(c.S1, cache, n+2)

    S2nm1 = c.get(c.S2, cache, n-1)
    S2n = c.get(c.S2, cache, n)
    S2np1 = c.get(c.S2, cache, n+1)
    S2nm1 = c.get(c.S2, cache, n-1)
    S2np2 = c.get(c.S2, cache, n+2)
    S2nm2 = c.get(c.S2, cache, n-2)

    S11nm1 = S1nm1 + 1./n

    if order==(1,0):

        res = ( 2. * (n**(-1) - (1. + n)**(-1)) ) / 3.  
        + (2. * ( S1n / n + S1np1 / (1. + n) ) ) / 3. 
        - (2. * ( -1./3. * np.pi**2 + S2nm1 + S2np1) ) / 3.
        + (4. * S11nm1 ) / 3.




    if order==(1,1):

        res = ( - 2. * ( - n**(-1) - (1. + n)**(-1)) ) / 3. 
        + (4. * S1nm1) / 3.


 
    if order==(1,2):
        
        res = 2./3. 

    return res

@nb.njit(cache=True)
def A_qQ2(n, order):

    if order==(1,0):

        res = 0.

    if order==(1,1):

        res = 0.


 
    if order==(1,2):
        
        res = 0.

    return res

@nb.njit(cache=True)
def A_qQ2bar(n, order):

    if order==(1,0):

        res = 0.

    if order==(1,1):

        res = 0.


 
    if order==(1,2):
        
        res = 0.
    
    return res

@nb.njit(cache=True)
def A_qqbar(n, order):

    if order==(1,0):

        res = 0.

    if order==(1,1):

        res = 0.


 
    if order==(1,2):
        
        res = 0.

    return res


@nb.njit(cache=True)
def A_entries(n, order, cache):
    r"""Compute the |NLO| singlet |OME|.

    .. math::
        A^{S,(1)} = \left(\begin{array}{cc}
        A_{gg,H}^{S,(1)} & 0  & A_{gH}^{(1)} \\
        0 & 0 & 0 \\
        A_{hg}^{S,(1)} & 0 & A_{HH}^{(1)}
        \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    numpy.ndarray
        |NLO| singlet |OME| :math:`A^{S,(1)}`

    """

    Agg = A_gg(n,order, cache)
    Agq = A_gq(n, order, cache)
    Aqg = A_qg(n, order, cache)
    Aqq = A_qq(n, order, cache)
    AqQ2 = A_qQ2(n, order) 
    Aqqbar = A_qqbar(n, order)
    AqQ2bar = A_qQ2bar(n, order)

    A_S = np.array(
        [
            [Agg, Agq, Agq, Agq, Agq],
            [Aqg, Aqq, Aqqbar, AqQ2, AqQ2bar],
            [Aqg, Aqqbar, Aqq, AqQ2bar, AqQ2],
            [Aqg, AqQ2, AqQ2bar, Aqq, Aqqbar],
            [Aqg, AqQ2bar, AqQ2bar, Aqqbar, Aqq],

        ], 
        
        np.complex_,
    )
    return A_S