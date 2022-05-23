# -*- coding: utf-8 -*-
# Test NNLO anomalous dims
import numpy as np

from eko.anomalous_dimensions.as4 import gNSm, gNSp, gNSv

# TODO: move this method  out of matching conditions if it will be used
# also here...
from eko.matching_conditions.operator_matrix_element import compute_harmonics_cache

NF = 5


def test_quark_number_conservation():
    N = 1
    sx_cache = compute_harmonics_cache(N, 3, False)

    # (ns,s)
    # the exact expression (nf^2 part) has an nonphysical pole at N=1,
    # see also :cite:`Moch:2017uml` and :cite:`Davies:2016jie` eq 3.5
    # where the \nu term is present.
    # This should cancel when doing the limit, since the given moment for N=1 is 0
    # np.testing.assert_allclose(gamma_nsv(N, NF, sx_cache), 0, rtol=3e-7)

    # nf^1 part
    np.testing.assert_allclose(gNSv.gamma_nss_nf1(N, sx_cache), 0.000400625, atol=2e-6)

    # (ns,-)
    # nf^3 part
    np.testing.assert_allclose(gNSp.gamma_ns_nf3(N, sx_cache), 0, atol=3e-15)
    # nf^2 part
    np.testing.assert_allclose(gNSm.gamma_nsm_nf2(N, sx_cache), 0, atol=9e-6)
    # nf^1 part
    np.testing.assert_allclose(gNSm.gamma_nsm_nf1(N, sx_cache), 0, atol=3e-7)
    # nf^0 part
    np.testing.assert_allclose(gNSm.gamma_nsm_nf0(N, sx_cache), 0, atol=7e-6)
    # total
    np.testing.assert_allclose(gNSm.gamma_nsm(N, NF, sx_cache), 0, atol=6e-6)


def test_non_singlet_reference_moments():

    NF = 4
    nsm_nf4_refs = [
        4322.890485339998,
        5491.581109692005,
        6221.256799360004,
        6774.606221595994,
        7229.056043916002,
        7618.358743427995,
        7960.658678124,
    ]
    nss_nf4_refs = [
        50.10532524,
        39.001939964,
        21.141505811200002,
        12.4834195012,
        8.0006134908,
        5.4610639744,
        3.9114290952,
        2.90857799,
    ]
    for N in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]:
        sx_cache = compute_harmonics_cache(N, 3, False)
        if N != 17:
            np.testing.assert_allclose(
                gNSm.gamma_nsm(N, NF, sx_cache), nsm_nf4_refs[int((N - 3) / 2)]
            )
        gamma_nss = (
            gNSv.gamma_nss_nf1(N, sx_cache) * NF
            + gNSv.gamma_nss_nf2(N, sx_cache) * NF**2
        )
        np.testing.assert_allclose(gamma_nss, nss_nf4_refs[int((N - 3) / 2)], atol=4e-4)


def test_singlet_reference_moments():
    NF = 4
    nsp_nf4_refs = [
        3679.6690577439995,
        5066.339235808004,
        5908.005605364002,
        6522.700744595994,
        7016.383458928004,
        7433.340927783997,
        7796.397038483998,
        8119.044600816003,
    ]
    for N in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]:
        sx_cache = compute_harmonics_cache(N, 3, False)
        np.testing.assert_allclose(
            gNSp.gamma_nsp(N, NF, sx_cache), nsp_nf4_refs[int((N - 2) / 2)], rtol=9e-4
        )


def test_diff_pm_nf2():
    # exact values of g_ns,+ prop to nf^2, see eq. 2.12 of :cite:`Davies:2016jie`
    gns_p_nf2_ref = [
        -2.803840877914952,
        117.7814976940519,
        188.87171647391625,
        238.68679901047244,
        277.07930877088023,
        308.4837803273823,
        335.1177191046878,
        358.3007784502141,
        378.85106713860694,
        397.32916271288485,
        414.1262555360935,
        429.5329384542155,
        443.76746775104476,
        457.00052146069135,
        469.36681505646123,
        480.9754844895223,
        491.9157919184809,
        502.26193535832397,
        512.0762438997319,
        521.4115529054055,
        530.3131765723368,
        538.820124769626,
        546.9664132606869,
        554.7816939160562,
        562.292174724311,
        569.5209392368877,
        576.488620836123,
        583.2135523833766,
        589.7122772356583,
        595.9996044443682,
        602.0890065401921,
        607.9926219826734,
        613.721571594659,
        619.2859319575805,
        624.6949916802943,
        629.9572094569793,
        635.0804247493609,
        640.0718085348396,
        644.9380387370721,
        649.6852483728882,
        654.3191732453407,
        658.8451002269538,
        663.2679927985199,
        667.592441020625,
        671.8227691456708,
        675.9629881109798,
        680.0168884780201,
        683.9879958439881,
        687.8796516493836,
        691.694971641426,
    ]
    n_init = 1
    for N in range(n_init, 51):
        if N == 2:
            rtol = 2e-3
        else:
            rtol = 1e-4
        sx_cache = compute_harmonics_cache(N, 3, False)
        np.testing.assert_allclose(
            gNSp.gamma_nsp_nf2(N, sx_cache), gns_p_nf2_ref[N - 1], rtol=rtol
        )
