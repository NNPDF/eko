# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.evolution_operator import OperatorMember

class TestOperatorMember:
    def test_name(self):
        shape = (2,2)
        ma,mae = np.random.rand(2,*shape)
        for i in ["S","g"]:
            for t in ["S","g"]:
                n = f"{t}.{i}"
                a = OperatorMember(ma,mae,n)
                # splitting
                assert a.input == i
                assert a.target == t
                # __str__
                assert str(a) == n
        # check wrongs
        for n in [".","a.",".b","ab"]:
            with pytest.raises(ValueError):
                o = OperatorMember(ma,mae,n)
                _t = o.target
                _i = o.input

    def test_add(self):
        shape = (2,2)
        ma,mb,mae,mbe = np.random.rand(4,*shape)

        # check plain sum
        a = OperatorMember(ma,mae,"S.S")
        b = OperatorMember(mb,mbe,"S.S")
        c = a + b

        assert_almost_equal(c.value, ma+mb)
        assert_almost_equal(c.error, mae+mbe)
        assert c.name == "S.S"

        # non matching name
        with pytest.raises(ValueError):
            _ = a + OperatorMember(mb,mbe,"S.g")

        # add to 0
        c = 0 + a
        assert_almost_equal(c.value, ma)
        assert_almost_equal(c.error, mae)
        assert c.name == "S.S"

        # add to 1
        with pytest.raises(ValueError):
            _ = 1 + OperatorMember(mb,mbe,"S.g")
