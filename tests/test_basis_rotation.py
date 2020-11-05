# -*- coding: utf-8 -*-
import pytest
import numpy as np
import eko.basis_rotation as br

n_x = 2


def test_rotate_pm_to_flavor():
    # g is still there
    assert all(([0] * (1 + 6) + [1] + [0] * 6) == br.rotate_pm_to_flavor("g"))
    # now t+ and t- are easiest
    assert all(([0] + [1] + [0] * (2 * 5 + 1) + [1]) == br.rotate_pm_to_flavor("t+"))
    assert all(([0] + [-1] + [0] * (2 * 5 + 1) + [1]) == br.rotate_pm_to_flavor("t-"))
    with pytest.raises(ValueError):
        br.rotate_pm_to_flavor("cbar")


class Test_fill_trivial_dists:
    def test_all(self):
        # all trivial
        evols, trival_dists = br.fill_trivial_dists(
            {l: np.random.rand(n_x) for l in ["g", "S", "V", "V3", "T3"]}
        )
        for q in [8, 15, 24, 35]:
            np.testing.assert_almost_equal(evols[f"V{q}"], evols["V"])
            assert f"V{q}" in trival_dists
            np.testing.assert_almost_equal(evols[f"T{q}"], evols["S"])
            assert f"T{q}" in trival_dists
        assert "ph" in trival_dists
        np.testing.assert_almost_equal(evols["ph"], np.zeros(n_x))

    def test_all_but_V8(self):
        # all but V8 trivial
        evols, trival_dists = br.fill_trivial_dists(
            {l: np.random.rand(n_x) for l in ["g", "S", "V", "V3", "T3", "T8"]}
        )
        np.testing.assert_almost_equal(evols["V8"], evols["V"])
        assert "V8" in trival_dists
        for q in [15, 24, 35]:
            np.testing.assert_almost_equal(evols[f"V{q}"], evols["V"])
            assert f"V{q}" in trival_dists
            np.testing.assert_almost_equal(evols[f"T{q}"], evols["S"])
            assert f"T{q}" in trival_dists
        assert "ph" in trival_dists
        np.testing.assert_almost_equal(evols["ph"], np.zeros(n_x))

    def test_errors(self):
        # errors
        with pytest.raises(KeyError, match="No S"):
            br.fill_trivial_dists({})
        with pytest.raises(KeyError, match="No V"):
            br.fill_trivial_dists({l: np.random.rand(n_x) for l in ["S", "V3", "T3"]})


class Test_rotate_output:
    def test_no_s(self):
        evols = {l: np.random.rand(n_x) for l in ["g", "S", "V", "V3", "T3"]}
        pdfs = br.rotate_output(evols)
        assert sorted(list(pdfs.keys())) == sorted([-3, -2, -1, 21, 1, 2, 3])
        np.testing.assert_allclose(evols["g"], pdfs[21])
        np.testing.assert_almost_equal(pdfs[3], np.zeros(n_x))
        np.testing.assert_almost_equal(pdfs[-3], np.zeros(n_x))
        np.testing.assert_allclose(evols["S"], pdfs[1] + pdfs[-1] + pdfs[2] + pdfs[-2])
        np.testing.assert_allclose(evols["V"], pdfs[1] - pdfs[-1] + pdfs[2] - pdfs[-2])
        np.testing.assert_allclose(
            evols["V3"], pdfs[2] - pdfs[-2] - (pdfs[1] - pdfs[-1])
        )
        np.testing.assert_allclose(
            evols["T3"], pdfs[2] + pdfs[-2] - (pdfs[1] + pdfs[-1])
        )

    def test_no_sv(self):
        evols = {l: np.random.rand(n_x) for l in ["g", "S", "V", "V3", "T3", "T8"]}
        pdfs = br.rotate_output(evols)
        assert sorted(list(pdfs.keys())) == sorted([-3, -2, -1, 21, 1, 2, 3])
        np.testing.assert_allclose(evols["g"], pdfs[21])
        np.testing.assert_almost_equal(pdfs[3], pdfs[-3])
        np.testing.assert_allclose(
            evols["S"], pdfs[1] + pdfs[-1] + pdfs[2] + pdfs[-2] + pdfs[3] + pdfs[-3]
        )
        np.testing.assert_allclose(evols["V"], pdfs[1] - pdfs[-1] + pdfs[2] - pdfs[-2])
        np.testing.assert_allclose(
            evols["V3"], pdfs[2] - pdfs[-2] - (pdfs[1] - pdfs[-1])
        )
        np.testing.assert_allclose(
            evols["T3"], pdfs[2] + pdfs[-2] - (pdfs[1] + pdfs[-1])
        )
        np.testing.assert_allclose(
            evols["T8"],
            pdfs[2] + pdfs[-2] + (pdfs[1] + pdfs[-1]) - 2 * (pdfs[3] + pdfs[-3]),
        )

    def test_no_sv_cv(self):
        evols = {
            l: np.random.rand(n_x) for l in ["g", "S", "V", "V3", "T3", "T8", "T15"]
        }
        pdfs = br.rotate_output(evols)
        assert sorted(list(pdfs.keys())) == sorted([-4, -3, -2, -1, 21, 1, 2, 3, 4])
        np.testing.assert_allclose(evols["g"], pdfs[21])
        np.testing.assert_almost_equal(pdfs[3], pdfs[-3])
        np.testing.assert_almost_equal(pdfs[4], pdfs[-4])
        np.testing.assert_allclose(
            evols["S"],
            (
                (pdfs[1] + pdfs[-1])
                + (pdfs[2] + pdfs[-2])
                + (pdfs[3] + pdfs[-3])
                + (pdfs[4] + pdfs[-4])
            ),
        )
        np.testing.assert_allclose(evols["V"], pdfs[1] - pdfs[-1] + pdfs[2] - pdfs[-2])
        np.testing.assert_allclose(
            evols["V3"], pdfs[2] - pdfs[-2] - (pdfs[1] - pdfs[-1])
        )
        np.testing.assert_allclose(
            evols["T3"], pdfs[2] + pdfs[-2] - (pdfs[1] + pdfs[-1])
        )
        np.testing.assert_allclose(
            evols["T8"],
            pdfs[2] + pdfs[-2] + (pdfs[1] + pdfs[-1]) - 2 * (pdfs[3] + pdfs[-3]),
        )
        np.testing.assert_allclose(
            evols["T15"],
            (pdfs[2] + pdfs[-2])
            + (pdfs[1] + pdfs[-1])
            + (pdfs[3] + pdfs[-3])
            - 3 * (pdfs[4] + pdfs[-4]),
        )


# fake PDFs - they only differ by the norm
fake_norms = {21: np.pi}
for q in range(1, 6 + 1):
    fake_norms[q] = q * 1.1
    fake_norms[-q] = float(q)


class FakePDF:
    def __init__(self, max_q):
        self.max_q = max_q

    def hasFlavor(self, pid):
        return pid in [21] + list(range(-self.max_q, self.max_q + 1))

    def xfxQ2(self, pid, x, q2):
        return fake_norms[pid] * x ** 2 * q2  # beware: its x*f(x,Q2)


class Test_generate_input_from_lhapdf:
    def test_no_s(self):
        pdf = FakePDF(2)
        for q2 in [1, 2, 3]:
            for xs in [np.array([1]), np.random.rand(5)]:
                inp = br.generate_input_from_lhapdf(pdf, xs, q2)
                assert sorted(list(inp.keys())) == sorted(
                    ["ph", "g", "S", "V", "V3", "T3"]
                )
                np.testing.assert_allclose(inp["ph"], 0 * xs * q2)
                np.testing.assert_allclose(inp["g"], np.pi * xs * q2)
                np.testing.assert_allclose(inp["S"], (2.1 + 4.2) * xs * q2)
                np.testing.assert_allclose(inp["V"], (0.1 + 0.2) * xs * q2)
                np.testing.assert_allclose(inp["V3"], (0.2 - 0.1) * xs * q2)
                np.testing.assert_allclose(inp["T3"], (4.2 - 2.1) * xs * q2)

    def test_no_c(self):
        pdf = FakePDF(3)
        for q2 in [1, 2, 3]:
            for xs in [np.array([1]), np.random.rand(5)]:
                inp = br.generate_input_from_lhapdf(pdf, xs, q2)
                assert sorted(list(inp.keys())) == sorted(
                    ["ph", "g", "S", "V", "V3", "T3", "V8", "T8"]
                )
                np.testing.assert_allclose(inp["ph"], 0 * xs * q2)
                np.testing.assert_allclose(inp["g"], np.pi * xs * q2)
                np.testing.assert_allclose(inp["S"], (2.1 + 4.2 + 6.3) * xs * q2)
                np.testing.assert_allclose(inp["V"], (0.1 + 0.2 + 0.3) * xs * q2)
                np.testing.assert_allclose(inp["V3"], (0.2 - 0.1) * xs * q2)
                np.testing.assert_allclose(inp["T3"], (4.2 - 2.1) * xs * q2)
                np.testing.assert_allclose(inp["V8"], (0.2 + 0.1 - 2 * 0.3) * xs * q2)
                np.testing.assert_allclose(inp["T8"], (4.2 + 2.1 - 2 * 6.3) * xs * q2)
