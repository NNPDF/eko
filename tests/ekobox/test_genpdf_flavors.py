# -*- coding: utf-8 -*-

import numpy as np

from ekobox import genpdf


def benchmark_is_evolution():
    assert genpdf.flavors.is_evolution_labels(["V", "T3"])
    assert not genpdf.flavors.is_evolution_labels(["21", "2"])


def benchmark_is_pids():
    assert not genpdf.flavors.is_pid_labels(["V", "T3"])
    assert not genpdf.flavors.is_pid_labels(["35", "9"])
    assert not genpdf.flavors.is_pid_labels({})
    assert genpdf.flavors.is_pid_labels([21, 2])


def benchmark_flavors_pid_to_flavor():
    flavs = genpdf.flavors.pid_to_flavor([1, 2, 21, -3])
    for f in flavs:
        for g in flavs:
            if not np.allclose(f, g):
                assert f @ g == 0


def benchmark_flavors_evol_to_flavor():
    flavs = genpdf.flavors.evol_to_flavor(["S", "g", "T3", "V8"])
    for f in flavs:
        for g in flavs:
            if not np.allclose(f, g):
                assert f @ g == 0


def benchmark_flavors_evol_raw():
    blocks = [
        {
            "Q2grid": np.array([1, 2]),
            "xgrid": np.array([0.1, 1.0]),
            "pids": np.array([-1, 21, 1]),
            "data": np.array([[0.1, 0.2, 0.1]] * 4),
        }
    ]
    gonly = genpdf.flavors.project(blocks, genpdf.flavors.evol_to_flavor(["g"]))
    assert len(gonly) == 1
    np.testing.assert_allclose(
        gonly[0]["data"],
        np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4
        ),
    )
    Sonly = genpdf.flavors.project(blocks, genpdf.flavors.evol_to_flavor(["S"]))
    assert len(Sonly) == 1
    for i in [0, 1, 2, 3]:
        # g and gamma are zero
        np.testing.assert_allclose(Sonly[0]["data"][i][7], 0)
        np.testing.assert_allclose(Sonly[0]["data"][i][0], 0)
        # quark are all equal and equal to anti-quarks
        for pid in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
            np.testing.assert_allclose(Sonly[0]["data"][i][pid], Sonly[0]["data"][i][1])


def benchmark_flavors_evol_nodata():
    # try with a block without data
    blocks = [
        {
            "Q2grid": np.array([1, 2]),
            "xgrid": np.array([0.1, 1.0]),
            "pids": np.array([-1, 21, 1]),
            "data": np.array([]),
        },
        {
            "Q2grid": np.array([1, 2]),
            "xgrid": np.array([0.1, 1.0]),
            "pids": np.array([-1, 21, 1]),
            "data": np.array([[0.1, 0.2, 0.1]] * 4),
        },
    ]
    gonly = genpdf.flavors.project(blocks, genpdf.flavors.evol_to_flavor(["g"]))
    assert len(gonly) == 2
    np.testing.assert_allclose(
        gonly[1]["data"],
        np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4
        ),
    )
