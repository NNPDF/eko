import numpy as np

from ekore import harmonics as h

from . import get_harmonic


def test_reset():
    c = h.cache.reset()
    np.testing.assert_allclose(c, np.full(c.shape, np.nan))
    available_items = np.sum(
        [isinstance(val, int) for val in h.cache.__dict__.values()]
    )
    assert c.shape == (available_items,)


def test_get():
    """Test getter function."""
    N = np.random.rand() + 1j * np.random.rand()
    for is_singlet in [True, False]:
        c = h.cache.reset()
        # loop on cached names
        for idx_name, idx in h.cache.__dict__.items():
            if not isinstance(idx, int):
                continue
            h.cache.get(idx, c, N, is_singlet)
            # Sx((N-1)/2)
            if "mh" in idx_name:
                h_func = get_harmonic(idx_name[:2])
                ref_value = h_func((N - 1) / 2)
            # Sx((N+1)/2)
            elif "ph" in idx_name:
                h_func = get_harmonic(idx_name[:2])
                ref_value = h_func((N + 1) / 2)
            # Sx(N/2)
            elif idx_name.endswith("h"):
                h_func = get_harmonic(idx_name[:2])
                ref_value = h_func(N / 2)
            # Smx(N)
            elif "Sm" in idx_name and len(idx_name) == 3:
                h_func = get_harmonic(idx_name[::2])
                h_minus_func = get_harmonic(idx_name)
                ref_value = h_minus_func(
                    N, h_func(N), h_func((N - 1) / 2), h_func(N / 2), is_singlet
                )
            # mellin g3
            elif idx_name == "g3":
                ref_value = h.g_functions.mellin_g3(
                    N, h.cache.get(h.cache.S1, c, N, is_singlet)
                )
            elif idx_name == "g3p2":
                ref_value = h.g_functions.mellin_g3(
                    N + 2, h.cache.get(h.cache.S1p2, c, N, is_singlet)
                )
            # Sx
            elif len(idx_name) == 2:
                h_func = get_harmonic(idx_name)
                ref_value = h_func(N)
            # all the others
            S1 = h.S1(N)
            S2 = h.S2(N)
            Sm1 = h.Sm1(N, S1, h.S1((N - 1) / 2), h.S1(N / 2), is_singlet)
            Sm2 = h.Sm2(N, S2, h.S2((N - 1) / 2), h.S2(N / 2), is_singlet)
            if idx_name == "S21":
                ref_value = h.S21(N, S1, S2)
            elif idx_name == "S2m1":
                ref_value = h.S2m1(N, S2, Sm1, Sm2, is_singlet)
            elif idx_name == "Sm21":
                ref_value = h.Sm21(N, S1, Sm1, is_singlet)
            elif idx_name == "Sm2m1":
                ref_value = h.Sm2m1(N, S1, S2, Sm2)
            elif idx_name == "S31":
                ref_value = h.S31(N, S1, S2, h.S3(N), h.S4(N))
            elif idx_name == "Sm31":
                ref_value = h.Sm31(N, S1, Sm1, Sm2, is_singlet)
            elif idx_name == "Sm22":
                Sm31 = h.Sm31(N, S1, Sm1, Sm2, is_singlet)
                ref_value = h.Sm22(N, S1, S2, Sm2, Sm31, is_singlet)
            elif idx_name == "S211":
                ref_value = h.S211(N, S1, S2, h.S3(N))
            elif idx_name == "Sm211":
                ref_value = h.Sm211(N, S1, S2, Sm1, is_singlet)
            elif idx_name == "S1p2":
                ref_value = h.S1(N + 2)
            np.testing.assert_allclose(c[idx], ref_value, err_msg=f"{idx_name}")
