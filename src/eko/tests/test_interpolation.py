# -*- coding: utf-8 -*-
# Test interpolation
from eko import t_float
from eko.interpolation import get_xgrid_linear_at_id, get_xgrid_linear_at_log

# for the numeric comparision to work, keep in mind that in Python3 the default precision is
# np.float64

def test_get_xgrid_linear_at_id():
    """test linear@id grids"""
    result = get_xgrid_linear_at_id(3,0.)
    assert all([isinstance(e,t_float) for e in result])
    assert all([a == b for a, b in zip(result, [0.,.5,1.])])

def test_get_xgrid_linear_at_log():
    """test linear@log grids"""
    result = get_xgrid_linear_at_log(3,1e-2)
    print(result)
    assert all([isinstance(e,t_float) for e in result])
    assert all([a == b for a, b in zip(result, [1e-2,1e-1,1.])])
