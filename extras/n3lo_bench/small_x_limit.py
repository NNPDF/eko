import msht_n3lo as msht
import numpy as np

pifact = (4 * np.pi) ** 4


def xpgg_to_0(x, nf):
    return -x * (
        (106911.99053742114 * np.log(x) ** 2) / x
        + (996.3830436187579 * nf * np.log(x) ** 2) / x
        + (8308.617314639116 * np.log(x) ** 3) / x
    )


def xpqg_to_0(x, nf):
    return -x * (-(3935.7613271019272 * nf * np.log(x) ** 2) / x)


def xpgq_to_0(x):
    return -x * (3692.7188065062737 * np.log(x) ** 3) / x


def xpqq_ps_to_0(x, nf):
    return -x * (-(1749.2272564897455 * nf * np.log(x) ** 2) / x)


def test_with_msht(myvals, mshtfunc, xgrid, nf=None):
    ref = np.array([x * mshtfunc(x) for x in xgrid])
    if nf is not None:
        np.testing.assert_allclose(myvals, nf * pifact * ref, rtol=6e-6)
    else:
        np.testing.assert_allclose(myvals, pifact * ref, rtol=4e-5)


def singlet_to_0(entry, x, nf):
    if entry == "gg":
        mysmall_x = xpgg_to_0(x, nf)
        test_with_msht(mysmall_x, msht.Pgg3exactA, x)
        return mysmall_x
    if entry == "gq":
        mysmall_x = xpgq_to_0(x)
        test_with_msht(mysmall_x, msht.Pgq3exactA, x)
        return mysmall_x
    if entry == "qg":
        mysmall_x = xpqg_to_0(x, nf)
        test_with_msht(mysmall_x, msht.Pqg3exactA, x, nf)
        return mysmall_x
    if entry == "qq":
        mysmall_x = xpqq_ps_to_0(x, nf)
        test_with_msht(mysmall_x, msht.Pqqps3exactA, x, nf)
        return mysmall_x
    return ValueError(f"{entry} not found")
