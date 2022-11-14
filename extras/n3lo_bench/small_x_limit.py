import numpy as np


def xpgg_to_0(x, nf):
    return - x * (
        (106911.99053742114 * np.log(x) ** 2) / x
        + (996.3830436187579 * nf * np.log(x) ** 2) / x
        + (8308.617314639116 * np.log(x) ** 3) / x
    )


def xpqg_to_0(x, nf):
    return - x * (-(3935.7613271019272 * nf * np.log(x) ** 2) / x)


def xpgq_to_0(x):
    return - x * (3692.7188065062737 * np.log(x) ** 3) / x


def xpqq_ps_to_0(x, nf):
    return - x * (-(1749.2272564897455 * nf * np.log(x) ** 2) / x)


def singlet_to_0(entry, x, nf):
    if entry == "gg":
        return xpgg_to_0(x, nf)
    if entry == "gq":
        return xpgq_to_0(x)
    if entry == "qg":
        return xpqg_to_0(x, nf)
    if entry == "qq":
        return xpqq_ps_to_0(x, nf)
    return ValueError(f"{entry} not found")
