# -*- coding: utf-8 -*-
import pytest

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
