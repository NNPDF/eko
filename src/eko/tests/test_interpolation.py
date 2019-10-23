# -*- coding: utf-8 -*-
# Test interpolation
import numpy as np
from numpy.polynomial import Chebyshev as T
import scipy.integrate as integrate

from eko import t_float
from eko.interpolation import get_xgrid_linear_at_id,get_xgrid_linear_at_log, \
    get_xgrid_Chebyshev_at_id, \
    get_Lagrange_iterpolators_x, get_Lagrange_iterpolators_N, \
    get_Lagrange_iterpolators_log_x, get_Lagrange_iterpolators_log_N

# for the numeric comparision to work, keep in mind that in Python3 the default precision is
# np.float64

def test_get_xgrid_linear_at_id():
    """test linear@id grids"""
    result = get_xgrid_linear_at_id(3,0.)
    assert all([isinstance(e,t_float) for e in result])
    assert all([a == b for a, b in zip(result, [0.,.5,1.])])

def test_get_xgrid_linear_at_log10():
    """test linear@log10 grids"""
    result = get_xgrid_linear_at_log(3,1e-2)
    assert all([isinstance(e,t_float) for e in result])
    assert all([a == b for a, b in zip(result, [1e-2,1e-1,1.])])

def test_get_xgrid_Chebyshev_at_id():
    """test get_xgrid_Chebyshev_at_id"""
    for n in [3,5,7]:
        result = get_xgrid_Chebyshev_at_id(n)
        assert all([isinstance(e,t_float) for e in result])
        # test that grid points correspond indeed to nodes of the polynomial
        assert all([np.abs(0. - T(np.append(np.zeros(n),1))(-1.+2.*e)) < 1e-10 for e in result])

def _Mellin_transform(f,N):
    """straight implementation of the Mellin transform"""
    r,re = integrate.quad(lambda x,f=f,N=N: np.real(x**(N-1)*f(x)),0,1)
    i,ie = integrate.quad(lambda x,f=f,N=N: np.imag(x**(N-1)*f(x)),0,1)
    return np.complex(r,i),np.complex(re,ie)

def test__Mellin_transform():
    """prevent circular reasoning"""
    f = lambda x : x
    g = lambda N : 1./(N+1.)
    for N in [1.,1.+1j,.5-2j]:
        e = g(N)
        a = _Mellin_transform(f,N)
        assert np.abs(e-a[0]) < 1e-6
        assert np.abs(a[1]) < 1e-6

#def test__special_beta():
#    """test reimplementation of beta"""
#    # built in only works on real arguments
#    for ab in [(1,1),(1,2),(2,1),(np.pi,np.pi)]:
#        a,b = ab
#        actual = _special_beta(a,b)
#        # ??? why does complain pylint ???
#        expected = special.beta(a,b) # pylint: disable=no-member
#        assert np.abs(actual - expected) < 1e-6

def test_f_xN():
    """test correspondence of interpolators in x- and N-space"""
    for fxfNg in [(get_Lagrange_iterpolators_x,    get_Lagrange_iterpolators_N,    [0., .5, 1.]),
                  (get_Lagrange_iterpolators_log_x,get_Lagrange_iterpolators_log_N,[1e-4,1e-2,1.0])
                 ]:
        fx,fN,g = fxfNg
        l = len(g)
        for j in range(l):
            for N in [1.,1.+1j,.5-2j]:
                a = fN(N,g,j)
                e = _Mellin_transform(lambda y,fx=fx,g=g,j=j:fx(y,g,j),N)
                assert np.abs(a-e[0]) < 1e-6
                assert np.abs(e[1]) < 1e-6

def test_is_iterpolators_x():
    """test that functions are indeed interpolators"""
    for fg in [(get_Lagrange_iterpolators_x,[0., .5, 1.]),
               (get_Lagrange_iterpolators_log_x,[1e-4,1e-2,1.0])
              ]:
        f,g = fg
        l = len(g)
        # sum needs to be one
        for x in [1e-4, 1e-2, .2, .4, .6, .8]:
            s = np.sum([f(x,g,j) for j in range(l)])
            assert np.abs(1.-s) < 1e-6
        # polynoms need to be "orthogonal" at grid points
        for j in range(l):
            one = f(g[j],g,j)
            assert np.abs(1.-one) < 1e-6
            for k in range(l):
                if j == k:
                    continue
                zero = f(g[k],g,j)
                assert np.abs(0.-zero) < 1e-6
