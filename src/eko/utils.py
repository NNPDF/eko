# -*- coding: utf-8 -*-
"""
    This file contains several utility functions
"""

import numpy as np


# https://stackoverflow.com/a/7205107
# from functools import reduce
# reduce(merge, [dict1, dict2, dict3...])
def merge_dicts(a: dict, b: dict, path=None):
    """
        Merges b into a.

        Parameters
        ----------
            a : dict
                target dictionary (modified)
            b : dict
                update
            path : array
                recursion track

        Returns
        -------
            a : dict
                updated dictionary
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def get_singlet_paths(to, fromm, depth):
    """
        Compute all possible path in the singlet sector to reach `to` starting from  `fromm`.

        Parameters
        ----------
            to : 'q' or 'g'
                final point
            fromm : 'q' or 'g'
                starting point
            depth : int
                nesting level; 1 corresponds to the trivial first step

        Returns
        -------
            ls : list
                list of all possible paths, where each path is in increasing order, e.g.
                [P1(c <- a), P2(c <- a), ...] and P1(c <- a) = [(c <- b), (b <- a)]
    """
    if depth < 1:
        raise ValueError(f"Invalid arguments: depth >= 1, but got {depth}")
    if to not in ["q", "g"]:
        raise ValueError(f"Invalid arguments: to in [q,g], but got {to}")
    if fromm not in ["q", "g"]:
        raise ValueError(f"Invalid arguments: fromm in [q,g], but got {fromm}")
    # trivial?
    if depth == 1:
        return [[f"S_{to}{fromm}"]]
    # do recursion (if necessary, we could switch back to loops instead)
    qs = get_singlet_paths(to, "q", depth - 1)
    for q in qs:
        q.append(f"S_q{fromm}")
    gs = get_singlet_paths(to, "g", depth - 1)
    for g in gs:
        g.append(f"S_g{fromm}")
    return qs + gs


def operator_product_helper(steps, paths):
    """
        Joins all matrix elements given by paths.

        Parameters
        ----------
            steps : array
                list of evolution steps in decreasing order, e.g.
                [..., O(c <- b), O(b <- a)]
            paths : array
                list of all possible paths: [P1(c <- a), P2(c <- a), ...]

        Returns
        -------
            tot_op : array
                joined operator
            tot_op_err : array
                combined error for operator
    """
    # setup
    len_steps = len(steps)
    # collect all paths
    tot_op = 0
    tot_op_err = 0
    for path in paths:
        # init product
        cur_op = None
        cur_op_err = None
        # check length
        if len(path) != len_steps:
            raise ValueError(
                "Number of steps and number of elements in a path do not match!"
            )
        # iterate steps
        for k, el in enumerate(path):
            new_op = steps[k]["operators"][el]
            new_op_err = steps[k]["operator_errors"][el]
            if cur_op is None:
                cur_op = new_op
                cur_op_err = new_op_err
            else:
                old_op = cur_op.copy()  # make copy for error determination
                cur_op = np.matmul(cur_op, new_op)
                cur_op_err = np.matmul(old_op, new_op_err) + np.matmul(
                    cur_op_err, new_op
                )
        # add up
        tot_op += cur_op
        tot_op_err += cur_op_err

    return tot_op, tot_op_err
