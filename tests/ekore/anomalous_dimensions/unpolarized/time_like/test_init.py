import eko.basis_rotation as br
import ekore.anomalous_dimensions.unpolarized.time_like as ad


def test_shapes():
    for k in range(1, 3 + 1):
        for s in ["+", "-", "V"]:
            assert ad.gamma_ns(
                (k, 0), br.non_singlet_pids_map[f"ns{s}"], 2.0, 5
            ).shape == (k,)
        assert ad.gamma_singlet((k, 0), 2.0, 5).shape == (k, 2, 2)
