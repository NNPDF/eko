"""Checks that the operator grid works as intended.

These test can be slow as they require the computation of several values
of Q But they should be fast as the grid is very small. It does *not*
test whether the result is correct, it can just test that it is sane
"""


# from eko.runner import legacy

# def test_compute_mu2grid(theory_ffns, operator_card, tmp_path):
#     mugrid = [(10.0, 5), (100.0, 5)]
#     operator_card.mugrid = mugrid
#     opgrid = legacy.Runner(
#         theory_ffns(3), operator_card, path=tmp_path / "eko.tar"
#     ).op_grid
#     opg = opgrid.compute()
#     assert len(opg) == len(mugrid)
#     assert all(k in op for k in ["operator", "error"] for op in opg.values())


# def test_grid_computation_VFNS(theory_card, operator_card, tmp_path):
#     """Checks that the grid can be computed."""
#     mugrid = [(3, 4), (5, 5), (5, 4)]
#     operator_card.mugrid = mugrid
#     opgrid = legacy.Runner(
#         theory_card, operator_card, path=tmp_path / "eko.tar"
#     ).op_grid
#     operators = opgrid.compute()
#     assert len(operators) == len(mugrid)


# def test_mod_expanded(theory_card, theory_ffns, operator_card, tmp_path: pathlib.Path):
#     mugrid = [(3, 4)]
#     operator_card.mugrid = mugrid
#     operator_card.configs.scvar_method = eko.io.types.ScaleVariationsMethod.EXPANDED
#     epsilon = 1e-1
#     path = tmp_path / "eko.tar"
#     for is_ffns, nf0 in zip([False, True], [5, 3]):
#         if is_ffns:
#             theory = theory_ffns(nf0)
#         else:
#             theory = theory_card
#         theory.order = (1, 0)
#         operator_card.init = (operator_card.init[0], nf0)
#         path.unlink(missing_ok=True)
#         opgrid = legacy.Runner(theory, operator_card, path=path).op_grid
#         opg = opgrid.compute()
#         theory.xif = 1.0 + epsilon
#         path.unlink(missing_ok=True)
#         sv_opgrid = legacy.Runner(theory, operator_card, path=path).op_grid
#         sv_opg = sv_opgrid.compute()
#         np.testing.assert_allclose(
#             opg[(9, 4)]["operator"], sv_opg[(9, 4)]["operator"], atol=2.5 * epsilon
#         )
