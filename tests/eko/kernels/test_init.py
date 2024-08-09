from eko.io.types import EvolutionMethod
from eko.kernels import EvoMethods, ev_method


def test_ev_method():
    methods = {
        "iterate-expanded": EvoMethods.ITERATE_EXPANDED,
        "decompose-expanded": EvoMethods.DECOMPOSE_EXPANDED,
        "perturbative-expanded": EvoMethods.PERTURBATIVE_EXPANDED,
        "truncated": EvoMethods.TRUNCATED,
        "ordered-truncated": EvoMethods.ORDERED_TRUNCATED,
        "iterate-exact": EvoMethods.ITERATE_EXACT,
        "decompose-exact": EvoMethods.DECOMPOSE_EXACT,
        "perturbative-exact": EvoMethods.PERTURBATIVE_EXACT,
    }
    assert len(methods.keys()) == len(EvolutionMethod)
    assert len(methods.keys()) == len(EvoMethods)
    for s, i in methods.items():
        j = ev_method(EvolutionMethod(s))
        assert j == i
        assert isinstance(j, int)
