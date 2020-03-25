"""
    Test constants module
"""

import pytest

from eko.constants import Constants

def test_str():
    c = Constants()
    expected = "{'NC': 3, 'TF': 0.5, 'CA': 3.0, 'CF': 1.3333333333333333}"
    assert str(c) == expected

def test_set():
    c = Constants()
    with pytest.raises(AttributeError):
        c.CA = 4
