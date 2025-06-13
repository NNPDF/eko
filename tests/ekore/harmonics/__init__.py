import numpy as np

from ekore import harmonics as h


def get_harmonic(name):
    for h_name, func in h.__dict__.items():
        if callable(func) and h_name == name:
            return func


def s(j, N):
    """S functions shortcut."""
    name = f"S{j}"
    h_func = get_harmonic(name)
    return h_func(N)


def sm(j, N, is_singlet=None):
    """Sm functions shortcut."""
    name = f"Sm{j}"
    h_func = get_harmonic(name[::2])
    h_minus_func = get_harmonic(name)
    return h_minus_func(N, h_func(N), h_func((N - 1) / 2), h_func(N / 2), is_singlet)


def sx(n, max_weight=5):
    """Get the harmonics sums S cache.

    Parameters
    ----------
    n : complex
        Mellin moment
    max_weight : int
        max harmonics weight, max value 5 (default)

    Returns
    -------
    np.ndarray
        list of harmonics sums (:math:`S_{1,..,w}`)
    """
    sx = np.zeros(max_weight, dtype=np.complex128)
    if max_weight >= 1:
        sx[0] = h.S1(n)
    if max_weight >= 2:
        sx[1] = h.S2(n)
    if max_weight >= 3:
        sx[2] = h.S3(n)
    if max_weight >= 4:
        sx[3] = h.S4(n)
    if max_weight >= 5:
        sx[4] = h.S5(n)
    return sx


def smx(n, sx, is_singlet):
    r"""Get the harmonics S-minus cache.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : numpy.ndarray
        List of harmonics sums: :math:`S_{1},\dots,S_{w}`
    is_singlet : bool
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
        False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    np.ndarray
        list of harmonics sums (:math:`S_{-1,..,-w}`)
    """
    max_weight = sx.size
    smx = np.zeros(max_weight, dtype=np.complex128)
    if max_weight >= 1:
        smx[0] = h.Sm1(n, sx[0], is_singlet)
    if max_weight >= 2:
        smx[1] = h.Sm2(n, sx[1], is_singlet)
    if max_weight >= 3:
        smx[2] = h.Sm3(n, sx[2], is_singlet)
    if max_weight >= 4:
        smx[3] = h.Sm4(n, sx[3], is_singlet)
    if max_weight >= 5:
        smx[4] = h.Sm5(n, sx[4], is_singlet)
