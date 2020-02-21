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

def operator_product(steps, list_of_paths):
    partial_op = []
    final_op = 0
    for path in list_of_paths:
        cur_op = None
        for step, member in zip(steps, path):
            new_op = step[member]
            if cur_op is None:
                cur_op = new_op
            else:
                cur_op = cur_op*new_op
        final_op += cur_op
    return final_op
