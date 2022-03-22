# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ekobox import genpdf


def test_genpdf_exceptions(tmp_path, cd):
    # using a wrong label and then a wrong parent pdf
    with cd(tmp_path):
        with pytest.raises(TypeError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions1",
                ["f"],
                {
                    21: lambda x, Q2: 3.0 * x * (1.0 - x),
                    2: lambda x, Q2: 4.0 * x * (1.0 - x),
                },
            )
        with pytest.raises(ValueError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions2",
                ["g"],
                10,
            )
        with pytest.raises(FileExistsError):
            genpdf.install_pdf("foo")

        with pytest.raises(TypeError):
            genpdf.generate_pdf("debug", [21], info_update=(10, 15, 20))


def test_generate_block():
    xg = np.linspace(0.0, 1.0, 5)
    q2s = np.geomspace(1.0, 1e3, 5)
    pids = np.arange(3)
    b = genpdf.generate_block(lambda pid, x, q2: pid * x * q2, xg, q2s, pids)
    assert isinstance(b, dict)
    assert sorted(b.keys()) == sorted(["data", "Q2grid", "xgrid", "pids"])
    assert isinstance(b["data"], np.ndarray)
    assert b["data"].shape == (len(xg) * len(q2s), len(pids))
