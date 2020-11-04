# -*- coding: utf-8 -*-

import numpy as np

import pytest

from eko import operator
from eko.operator import flavors


def mk_op_members(shape=(2, 2)):
    m = np.random.rand(7, *shape)
    om = {}
    for j, l in enumerate(operator.full_labels):
        om[l] = m[j]
    return om


def test_ad_to_evol_map_ffns():
    oms = mk_op_members()
    triv_ops = ("S.S", "S.g", "g.S", "g.g", "V.V", "V3.V3", "T3.T3", "V8.V8", "T8.T8")
    # FFNS3
    m = flavors.ad_to_evol_map(oms, 3, False)
    assert sorted(triv_ops) == sorted(m.keys())
    # FFNS3 + IC
    m = flavors.ad_to_evol_map(oms, 3, False, [4])
    assert sorted([*triv_ops, "c+.c+", "c-.c-"]) == sorted(m.keys())
    # FFNS3 + IC + IB
    m = flavors.ad_to_evol_map(oms, 3, False, [4, 5])
    assert sorted([*triv_ops, "c+.c+", "c-.c-", "b+.b+", "b-.b-"]) == sorted(m.keys())
    # FFNS4 + IC + IB
    with pytest.raises(ValueError):
        m = flavors.ad_to_evol_map(oms, 4, False, [4, 5])
    # FFNS4 + IB
    m = flavors.ad_to_evol_map(oms, 4, False, [5])
    assert sorted([*triv_ops, "V15.V15", "T15.T15", "b+.b+", "b-.b-"]) == sorted(
        m.keys()
    )
    # FFNS6
    m = flavors.ad_to_evol_map(oms, 6, False)
    assert len(m.keys()) == 4 + 1 + 2*5


def test_ad_to_evol_map_vfns():
    oms = mk_op_members()
    triv_ops = ("S.S", "S.g", "g.S", "g.g", "V.V", "V3.V3", "T3.T3", "V8.V8", "T8.T8")
    # VFNS, nf=3 patch
    m = flavors.ad_to_evol_map(oms, 3, True)
    assert sorted([*triv_ops, "V15.V", "T15.S", "T15.g"]) == sorted(m.keys())
    # VFNS, nf=3 patch + IC
    m = flavors.ad_to_evol_map(oms, 3, True, [4])
    assert sorted([*triv_ops, "V15.V", "T15.S", "T15.g", "T15.c+", "V15.c-"]) == sorted(
        m.keys()
    )
    # VFNS, nf=3 patch + IC + IB
    m = flavors.ad_to_evol_map(oms, 3, True, [4, 5])
    assert sorted(
        [*triv_ops, "V15.V", "T15.S", "T15.g", "T15.c+", "V15.c-", "b+.b+", "b-.b-"]
    ) == sorted(m.keys())
    # VFNS, nf=4 patch + IC + IB
    m = flavors.ad_to_evol_map(oms, 4, True, [4, 5])
    ks = sorted(
        [*triv_ops, "V15.V15", "T15.T15", "V24.V", "T24.S", "T24.g", "T24.b+", "V24.b-"]
    )
    assert ks == sorted(m.keys())
    # VFNS, nf=4 patch + IB
    m = flavors.ad_to_evol_map(oms, 4, True, [5])
    assert ks == sorted(m.keys())
