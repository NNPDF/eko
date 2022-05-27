# -*- coding: utf-8 -*-
from eko.compile import parse


def full__nklo():
    a = 1
    # >> start/N1LO
    a += 2
    # << end/N1LO
    # >> start/N2LO
    a += 4
    # << end/N2LO
    # >> start/N3LO
    a += 8
    # << end/N3LO
    return a


full__lo = parse(full__nklo, 0, globals())
full__nlo = parse(full__nklo, 1, globals())
full__nnlo = parse(full__nklo, 2, globals())
full__nnnlo = full__nklo


def chain__nklo():
    return -full__nklo()


chain__lo = parse(chain__nklo, 0, globals())
chain__nlo = parse(chain__nklo, 1, globals())
chain__nnlo = parse(chain__nklo, 2, globals())
chain__nnnlo = chain__nklo


def partial__nklo():
    a = "a"
    # >> start/N3LO
    a += "b"
    # << end/N3LO
    return a


partial__lo = parse(partial__nklo, 0, globals())
partial__nlo = parse(partial__nklo, 1, globals())
partial__nnlo = parse(partial__nklo, 2, globals())
partial__nnnlo = partial__nklo


def broken__nklo():
    a = "c"
    # >> start/N3LO
    a += "d"
    return a


broken__lo = parse(broken__nklo, 0, globals())
broken__nlo = parse(broken__nklo, 1, globals())
broken__nnlo = parse(broken__nklo, 2, globals())
broken__nnnlo = broken__nklo


def test_parse():
    # different at each order
    assert full__lo() == 1
    assert full__nlo() == 3
    assert full__nnlo() == 7
    assert full__nnnlo() == 15
    # chained call
    assert chain__lo() == -1
    assert chain__nlo() == -3
    assert chain__nnlo() == -7
    assert chain__nnnlo() == -15
    # only N3LO sensitive
    assert partial__lo() == "a"
    assert partial__nlo() == "a"
    assert partial__nnlo() == "a"
    assert partial__nnnlo() == "ab"
    # broken parsing (falling through to N3LO)
    assert broken__lo() == "cd"
    assert broken__nlo() == "cd"
    assert broken__nnlo() == "cd"
    assert broken__nnnlo() == "cd"
