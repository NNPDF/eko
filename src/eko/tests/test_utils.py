# -*- coding: utf-8 -*-
import numpy as np

import eko.utils as utils


def test_get_singlet_paths():
    # trivial solution
    a = utils.get_singlet_paths("q", "q", 1)
    assert a == [["S_qq"]]
    # level 2
    b = utils.get_singlet_paths("q", "g", 2)
    assert b == [["S_qq", "S_qg"], ["S_qg", "S_gg"]]
    # level 4
    c = utils.get_singlet_paths("g", "g", 4)
    assert len(c) == 2 ** 3
    for path in c:
        # check start + end
        assert path[0][-2] == "g"
        assert path[-1][-1] == "g"
        # check concatenation
        for k, el in enumerate(path[:-1]):
            assert el[-1] == path[k + 1][-2]


def test_operator_product_helper_1():
    # setup test: q from g
    paths = utils.get_singlet_paths("q", "g", 1)
    # i.e., in step 1, we can do g -> g or g -> q, as we start from g
    ls = np.random.rand(2, 2, 4)
    S_qg_1, S_qg_1_err = ls[0], ls[1]
    step1 = {"operators": {"S_qg": S_qg_1}, "operator_errors": {"S_qg": S_qg_1_err}}
    # setup target
    target_op = S_qg_1
    target_op_err = S_qg_1_err
    # run
    tot_op, tot_op_err = utils.operator_product_helper([step1], paths)
    # test
    np.testing.assert_array_equal(tot_op, target_op)
    np.testing.assert_array_equal(tot_op_err, target_op_err)


def test_operator_product_helper_2():
    # setup test: q from g via 1 step in between
    paths = utils.get_singlet_paths("q", "g", 2)
    # i.e., in step 1, we can do g <- g or q <- g, as we need to start from g
    ls1 = np.random.rand(4, 3, 4)
    S_gg_1, S_gg_1_err = ls1[0], ls1[1]
    S_qg_1, S_qg_1_err = ls1[2], ls1[3]
    step1 = {
        "operators": {"S_gg": S_gg_1, "S_qg": S_qg_1},
        "operator_errors": {"S_gg": S_gg_1_err, "S_qg": S_qg_1_err},
    }
    # then, in step 2, we do q <- q or q <- g, as we need to finish in q
    ls2 = np.random.rand(4, 2, 3)
    S_qg_2, S_qg_2_err = ls2[0], ls2[1]
    S_qq_2, S_qq_2_err = ls2[2], ls2[3]
    step2 = {
        "operators": {"S_qg": S_qg_2, "S_qq": S_qq_2},
        "operator_errors": {"S_qg": S_qg_2_err, "S_qq": S_qq_2_err},
    }
    # setup target
    target_op = np.matmul(S_qg_2, S_gg_1) + np.matmul(S_qq_2, S_qg_1)
    target_op_err = (
        np.matmul(S_qg_2, S_gg_1_err)
        + np.matmul(S_qg_2_err, S_gg_1)
        + np.matmul(S_qq_2_err, S_qg_1)
        + np.matmul(S_qq_2, S_qg_1_err)
    )
    # run
    tot_op, tot_op_err = utils.operator_product_helper([step2, step1], paths)
    # test
    np.testing.assert_allclose(tot_op, target_op)
    np.testing.assert_allclose(tot_op_err, target_op_err)
