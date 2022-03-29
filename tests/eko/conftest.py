# -*- coding: utf-8 -*-
import numpy as np
import pytest

from eko.anomalous_dimensions import harmonics


@pytest.fixture
def get_sx():
    def wrapped(N):
        """Collect the S-cache"""
        sx = np.array(
            [
                harmonics.S1(N),
                harmonics.S2(N),
                harmonics.S3(N),
                harmonics.S4(N),
                harmonics.S5(N),
            ]
        )
        return sx

    return wrapped
