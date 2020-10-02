# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.operator.operator import Operator
from eko import thresholds

# int_0.5^1 dz z^k = (1 - 0.5**(k+1))/(k+1)
def get_ker(k):
    def ker(z, args, k=k):
        lnx = args[0]
        return lnx * z ** k

    return ker


def get_res(k, cut=0):
    return [
        [x * ((1 - cut) ** (k + 1) - 0.5 ** (k + 1)) / (k + 1), 0]
        for x in np.log([0.5, 1.0])
    ]


# class TestOperator:
#     def test_compute_LO(self, mock_OpMaster):
#         master = mock_OpMaster(
#             [
#                 dict(
#                     NS_p=get_ker(1),
#                     S_qq=get_ker(2),
#                     S_qg=get_ker(3),
#                     S_gq=get_ker(4),
#                     S_gg=get_ker(5),
#                 )
#             ],
#             0,
#         )
#         op = Operator(master, 0.5, 1.0, 0)
#         op.compute()

#         assert_almost_equal(op.op_members["NS_p"].value, get_res(1))
#         assert_almost_equal(op.op_members["NS_m"].value, get_res(1))
#         assert_almost_equal(op.op_members["NS_v"].value, get_res(1))
#         assert_almost_equal(op.op_members["S_qq"].value, get_res(2))
#         assert_almost_equal(op.op_members["S_qg"].value, get_res(3))
#         assert_almost_equal(op.op_members["S_gq"].value, get_res(4))
#         assert_almost_equal(op.op_members["S_gg"].value, get_res(5))

#     def test_skip(self, mock_OpMaster):
#         master = mock_OpMaster(
#             [
#                 dict(
#                     NS_p=get_ker(1),
#                     S_qq=get_ker(2),
#                     S_qg=get_ker(3),
#                     S_gq=get_ker(4),
#                     S_gg=get_ker(5),
#                 )
#             ],
#             0,
#             True,
#         )
#         op = Operator(master, 0.5, 1.0, 0)
#         op.compute()

#         zero = np.zeros((2, 2))
#         assert_almost_equal(op.op_members["NS_p"].value, zero)
#         assert_almost_equal(op.op_members["NS_m"].value, zero)
#         assert_almost_equal(op.op_members["NS_v"].value, zero)
#         assert_almost_equal(op.op_members["S_qq"].value, zero)
#         assert_almost_equal(op.op_members["S_qg"].value, zero)
#         assert_almost_equal(op.op_members["S_gq"].value, zero)
#         assert_almost_equal(op.op_members["S_gg"].value, zero)

#     def test_compute_NLO(self, mock_OpMaster):
#         master = mock_OpMaster(
#             [
#                 dict(
#                     NS_p=get_ker(1),
#                     NS_m=get_ker(2),
#                     S_qq=get_ker(3),
#                     S_qg=get_ker(4),
#                     S_gq=get_ker(5),
#                     S_gg=get_ker(6),
#                 )
#             ],
#             1,
#         )
#         op = Operator(master, 0.5, 1.0, 0)
#         op.compute()

#         assert_almost_equal(op.op_members["NS_p"].value, get_res(1))
#         assert_almost_equal(op.op_members["NS_m"].value, get_res(2))
#         assert_almost_equal(op.op_members["NS_v"].value, get_res(2))
#         assert_almost_equal(op.op_members["S_qq"].value, get_res(3))
#         assert_almost_equal(op.op_members["S_qg"].value, get_res(4))
#         assert_almost_equal(op.op_members["S_gq"].value, get_res(5))
#         assert_almost_equal(op.op_members["S_gg"].value, get_res(6))

#     def test_compose(self, mock_OpMaster):
#         master = mock_OpMaster(
#             [
#                 dict(
#                     NS_p=get_ker(1),
#                     S_qq=get_ker(2),
#                     S_qg=get_ker(3),
#                     S_gq=get_ker(4),
#                     S_gg=get_ker(5),
#                 )
#             ],
#             0,
#         )
#         op1 = Operator(master, 0.5, 1.0, 0)
#         # FFNS
#         t = thresholds.ThresholdsConfig(1, "FFNS", nf=3)
#         instruction_set = t.get_composition_path(3, 0)
#         ph = op1.compose([], instruction_set, 2)
#         assert isinstance(ph, PhysicalOperator)
#         # V.V is NS_v
#         assert_almost_equal(ph.op_members["V.V"].value, get_res(1))
