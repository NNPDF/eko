import io
from dataclasses import dataclass
from typing import NewType, Sequence, Union

import numpy as np
import numpy.typing as npt
import pytest
import yaml

from eko import interpolation
from eko.io import dictlike


@dataclass
class PlainDataclass:
    i: int
    f: float


@dataclass
class MyNestedDictLike(dictlike.DictLike):
    pdc: PlainDataclass
    f: float


@dataclass
class MyDictLike(dictlike.DictLike):
    nda: npt.NDArray
    f: float
    x: interpolation.XGrid
    t: tuple
    s: str
    d: dict
    ndl: MyNestedDictLike


def test_serialization():
    d = MyDictLike.from_dict(
        dict(
            nda=np.arange(5.0),
            f=np.arange(5.0)[-1],
            x=[0.1, 1.0],
            t=(1.0, 2.0),
            s="s",
            d=dict(my="very", nice="dict"),
            ndl=dict(pdc=dict(i=10, f=1.61), f=3.14),
        )
    )
    assert d.f == 4.0
    dd = MyDictLike.from_dict(d.raw)
    assert dd.f == 4.0
    # check we can dump and reload
    stream = io.StringIO()
    yaml.safe_dump(d.raw, stream)
    stream.seek(0)
    ddd = yaml.safe_load(stream)
    assert "nda" in ddd
    np.testing.assert_allclose(ddd["nda"], np.arange(5.0))


@dataclass
class UnsupportedDictLike(dictlike.DictLike):
    sf: Sequence[float]


@dataclass
class UnsupportedUnionDictLike(dictlike.DictLike):
    uisf: Union[int, Sequence[float]]


def test_unsupported():
    """Abstract type hints.

    It is not possible to deserialize actually abstract type hints, not
    corresponding to any instantiatable class.
    """
    with pytest.raises(TypeError):
        UnsupportedDictLike.from_dict(dict(sf=[3.14]))
    with pytest.raises(TypeError):
        UnsupportedUnionDictLike.from_dict(dict(uisf=[3.14]))


@dataclass
class ListLike(dictlike.DictLike):
    a: int
    b: int
    c: int


def test_dictlike_list():
    ls = [1, 2, 3]
    lslike = ListLike.from_dict(ls)

    assert isinstance(lslike, ListLike)
    assert lslike.a == ls[0]
    assert lslike.c == ls[2]


newint = NewType("newint", int)


@dataclass
class WithNewType(dictlike.DictLike):
    nt: newint


def test_newtype():
    val = 42
    wnt = WithNewType.from_dict(dict(nt=val))

    assert wnt.nt == val
