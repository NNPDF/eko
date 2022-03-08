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
                harmonics.harmonic_S1(N),
                harmonics.harmonic_S2(N),
                harmonics.harmonic_S3(N),
                harmonics.harmonic_S4(N),
                harmonics.harmonic_S5(N),
            ]
        )
        return sx

    return wrapped
