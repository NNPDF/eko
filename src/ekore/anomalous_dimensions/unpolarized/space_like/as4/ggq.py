# pylint: skip-file
# fmt: off
r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{gq}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .....harmonics.log_functions import (
    lm11,
    lm11m1,
    lm11m2,
    lm12,
    lm12m1,
    lm12m2,
    lm13,
    lm13m1,
    lm13m2,
    lm14,
    lm14m1,
    lm14m2,
    lm15,
)


@nb.njit(cache=True)
def gamma_gq_nf3(n, cache):
    r"""Return the part proportional to :math:`nf^3` of :math:`\gamma_{gq}^{(3)}`.

    The expression is copied exact from :eqref:`3.13` of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^3}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    return 1.3333333333333333 * (
        -11.39728026699467 / (-1.0 + n)
        + 11.39728026699467 / n
        - 2.3703703703703702 / np.power(1.0 + n, 4)
        + 6.320987654320987 / np.power(1.0 + n, 3)
        - 3.1604938271604937 / np.power(1.0 + n, 2)
        - 5.698640133497335 / (1.0 + n)
        - (6.320987654320987 * S1) / (-1.0 + n)
        + (6.320987654320987 * S1) / n
        - (2.3703703703703702 * S1) / np.power(1.0 + n, 3)
        + (6.320987654320987 * S1) / np.power(1.0 + n, 2)
        - (3.1604938271604937 * S1) / (1.0 + n)
        + (6.320987654320987 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        - (6.320987654320987 * (np.power(S1, 2) + S2)) / n
        - (1.1851851851851851 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        + (3.1604938271604937 * (np.power(S1, 2) + S2)) / (1.0 + n)
        - (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        + (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        - (0.3950617283950617 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_gq_nf0(n, cache, variation):
    r"""Return the part proportional to :math:`nf^0` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^0}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    S5 = c.get(c.S5, cache, n)
    common = -22156.31283903764/np.power(-1. + n,4) + 95032.88047770769/np.power(-1. + n,3) - 37609.87654320987/np.power(n,7) - 35065.67901234568/np.power(n,6) - 175454.58483973087/np.power(n,5) - 375.3983146907502*lm14(n,S1,S2,S3,S4) - 13.443072702331962*lm15(n,S1,S2,S3,S4,S5)
    if variation == 1:
        fit = -135325.37409909506/np.power(-1. + n,2) + 107389.69725534944/(-1. + n) - 281247.1594541515/(1. + n) + 145447.60744097419/(2. + n) - 1644.0474725539857*lm13(n,S1,S2,S3)
    elif variation == 2:
        fit = -130157.18895885911/np.power(-1. + n,2) + 93282.05706699888/(-1. + n) - 169738.63893980338/(1. + n) + 9212.769374520116*lm12(n,S1,S2) - 223.51479074632925*lm13(n,S1,S2,S3)
    elif variation == 3:
        fit = -130767.27520270286/np.power(-1. + n,2) + 94891.36609978844/(-1. + n) - 212410.3233945088/(1. + n) - 32955.07417912945*lm11(n,S1) - 1046.7526219200904*lm13(n,S1,S2,S3)
    elif variation == 4:
        fit = -158783.450560317/np.power(-1. + n,2) + 90100.75345693348/(-1. + n) + 412018.22879229486/np.power(n,4) - 129027.27855213353/(1. + n) - 1753.8825871656047*lm13(n,S1,S2,S3)
    elif variation == 5:
        fit = -129693.48645695807/np.power(-1. + n,2) + 52876.75483885584/(-1. + n) + 158964.13837584556/np.power(n,3) - 85916.6315255873/(1. + n) - 1690.6720701417717*lm13(n,S1,S2,S3)
    elif variation == 6:
        fit = -301210.2328458577/np.power(-1. + n,2) + 811215.7479286905/(-1. + n) - 1.242177631711221e6/np.power(n,2) - 857543.1305664484/(1. + n) - 1833.163010136661*lm13(n,S1,S2,S3)
    elif variation == 7:
        fit = -122290.16043542989/np.power(-1. + n,2) + 71807.36048618097/(-1. + n) - 221400.82937350916/(2. + n) + 23236.477404056477*lm12(n,S1,S2) + 1938.8247584430924*lm13(n,S1,S2,S3)
    elif variation == 8:
        fit = -116702.31643837337/np.power(-1. + n,2) + 56325.17516760884/(-1. + n) - 448808.73529422894/(2. + n) - 134644.78516249143*lm11(n,S1) + 796.324422989612*lm13(n,S1,S2,S3)
    elif variation == 9:
        fit = -178667.39573180894/np.power(-1. + n,2) + 75445.99644274621/(-1. + n) + 761260.3281816904/np.power(n,4) - 123286.8456395018/(2. + n) - 1846.982948588938*lm13(n,S1,S2,S3)
    elif variation == 10:
        fit = -127216.28629540099/np.power(-1. + n,2) + 28899.09790707029/(-1. + n) + 228884.9205878143/np.power(n,3) - 63975.501562926234/(2. + n) - 1711.180018456112*lm13(n,S1,S2,S3)
    elif variation == 11:
        fit = -54369.324330163065/np.power(-1. + n,2) - 236095.39968148115/(-1. + n) + 606214.4244773745/np.power(n,2) + 216429.75635890095/(2. + n) - 1551.7542579210171*lm13(n,S1,S2,S3)
    elif variation == 12:
        fit = -127730.39915689365/np.power(-1. + n,2) + 86880.5772952685/(-1. + n) + 131088.0859004977*lm11(n,S1) + 45859.153375434624*lm12(n,S1,S2) + 3051.145461650118*lm13(n,S1,S2,S3)
    elif variation == 13:
        fit = -249509.19871481528/np.power(-1. + n,2) + 80018.18821849066/(-1. + n) + 1.7178353341095438e6/np.power(n,4) - 29198.20288498338*lm12(n,S1,S2) - 6604.105966476303*lm13(n,S1,S2,S3)
    elif variation == 14:
        fit = -129218.19651555142/np.power(-1. + n,2) + 11461.766456848798/(-1. + n) + 321900.6239594061/np.power(n,3) - 9442.986825272894*lm12(n,S1,S2) - 3194.4920722194884*lm13(n,S1,S2,S3)
    elif variation == 15:
        fit = -87944.16009766924/np.power(-1. + n,2) - 83891.96406756257/(-1. + n) + 306548.65313472337/np.power(n,2) + 11486.326691371696*lm12(n,S1,S2) + 173.71945424342616*lm13(n,S1,S2,S3)
    elif variation == 16:
        fit = -202135.7995740037/np.power(-1. + n,2) + 82687.73905965324/(-1. + n) + 1.0495769899938635e6/np.power(n,4) + 50994.82207509583*lm11(n,S1) - 2848.098505650866*lm13(n,S1,S2,S3)
    elif variation == 17:
        fit = -128964.1511783234/np.power(-1. + n,2) + 24339.728042797506/(-1. + n) + 266935.24034010764/np.power(n,3) + 22383.637660612447*lm11(n,S1) - 2128.032930969243*lm13(n,S1,S2,S3)
    elif variation == 18:
        fit = -74648.83981878728/np.power(-1. + n,2) - 140958.82150230306/(-1. + n) + 408987.65266967454/np.power(n,2) - 43805.55005962004*lm11(n,S1) - 787.8265790554562*lm13(n,S1,S2,S3)
    elif variation == 19:
        fit = -71719.13674652952/np.power(-1. + n,2) - 21308.180757420447/(-1. + n) - 821124.7287279929/np.power(n,4) + 475769.01709201955/np.power(n,3) - 1564.6977442093062*lm13(n,S1,S2,S3)
    elif variation == 20:
        fit = -133558.27521182265/np.power(-1. + n,2) - 37615.76088185934/(-1. + n) + 484990.68454355566/np.power(n,4) + 220001.80072250665/np.power(n,2) - 1739.8412493001551*lm13(n,S1,S2,S3)
    elif variation == 21:
        fit = -110595.9812283641/np.power(-1. + n,2) - 31560.381420415717/(-1. + n) + 176663.97543390834/np.power(n,3) + 138310.07359867013/np.power(n,2) - 1674.8064387740321*lm13(n,S1,S2,S3)
    else:
        fit = -138152.69664751078/np.power(-1. + n,2) + 57913.880829154245/(-1. + n) + 171645.5636615693/np.power(n,4) + 77577.04360900483/np.power(n,3) + 20851.665375796576/np.power(n,2) - 82661.10297298252/(1. + n) - 23599.740384299566/(2. + n) - 330.42208404928425*lm11(n,S1) + 2435.882720720316*lm12(n,S1,S2) - 1327.8017698551957*lm13(n,S1,S2,S3)
    return common + fit


@nb.njit(cache=True)
def gamma_gq_nf1(n, cache, variation):
    r"""Return the part proportional to :math:`nf^1` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^1}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    S5 = c.get(c.S5, cache, n)
    common = 885.6738165500071/np.power(-1. + n,3) + 5309.62962962963/np.power(n,7) + 221.23456790123456/np.power(n,6) + 9092.91243376357/np.power(n,5) + 34.49474165523548*lm14(n,S1,S2,S3,S4) + 0.5486968449931413*lm15(n,S1,S2,S3,S4,S5)
    if variation == 1:
        fit = -4154.154695948995/np.power(-1. + n,2) + 10568.669617295407/(-1. + n) - 17488.846844461823/(1. + n) + 12359.004664911017/(2. + n) + 211.10404507107253*lm13(n,S1,S2,S3)
    elif variation == 2:
        fit = -3715.00258213197/np.power(-1. + n,2) + 9369.912250526479/(-1. + n) - 8013.721406089578/(1. + n) + 782.8293753310119*lm12(n,S1,S2) + 331.8098466308406*lm13(n,S1,S2,S3)
    elif variation == 3:
        fit = -3766.8429586516354/np.power(-1. + n,2) + 9506.658795852165/(-1. + n) - 11639.62874491426/(1. + n) - 2800.265488570204*lm11(n,S1) + 261.8575086390502*lm13(n,S1,S2,S3)
    elif variation == 4:
        fit = -6147.4392986647545/np.power(-1. + n,2) + 9099.589851629416/(-1. + n) + 35010.09952148762/np.power(n,4) - 4554.387135843054/(1. + n) + 201.7711126307286*lm13(n,S1,S2,S3)
    elif variation == 5:
        fit = -3675.6007556945196/np.power(-1. + n,2) + 5936.584418528788/(-1. + n) + 13507.53417196719/np.power(n,3) - 891.1801771144223/(1. + n) + 207.1422499668006*lm13(n,S1,S2,S3)
    elif variation == 6:
        fit = -18249.757773649242/np.power(-1. + n,2) + 70374.32470231941/(-1. + n) - 105550.57876215602/np.power(n,2) - 66457.98920891003/(1. + n) + 195.03447986946037*lm13(n,S1,S2,S3)
    elif variation == 7:
        fit = -3343.5834681677507/np.power(-1. + n,2) + 8356.04624598052/(-1. + n) - 10452.80306687096/(2. + n) + 1444.9183960217892*lm12(n,S1,S2) + 433.89848620522463*lm13(n,S1,S2,S3)
    elif variation == 8:
        fit = -2996.1134775555597/np.power(-1. + n,2) + 7393.314503624469/(-1. + n) - 24593.753132227568/(2. + n) - 8372.642876397704*lm11(n,S1) + 362.8541655256052*lm13(n,S1,S2,S3)
    elif variation == 9:
        fit = -6849.300061096075/np.power(-1. + n,2) + 8582.308253713018/(-1. + n) + 47337.59912160785/np.power(n,4) - 4351.762124258069/(2. + n) + 198.4848688754035*lm13(n,S1,S2,S3)
    elif variation == 10:
        fit = -3649.905711097048/np.power(-1. + n,2) + 5687.873404469671/(-1. + n) + 14232.795555815104/np.power(n,3) - 663.5932741012635/(2. + n) + 206.92952890585326*lm13(n,S1,S2,S3)
    elif variation == 11:
        fit = 879.9511001376588/np.power(-1. + n,2) - 10790.331004577562/(-1. + n) + 37696.34952104189/np.power(n,2) + 16772.9014435428/(2. + n) + 216.84313199599924*lm13(n,S1,S2,S3)
    elif variation == 12:
        fit = -3600.42868484152/np.power(-1. + n,2) + 9067.684803683303/(-1. + n) + 6188.946763244337*lm11(n,S1) + 2512.9829441244183*lm12(n,S1,S2) + 486.41350744844374*lm13(n,S1,S2,S3)
    elif variation == 13:
        fit = -9349.864281360744/np.power(-1. + n,2) + 8743.696850292556/(-1. + n) + 81102.65214258479/np.power(n,4) - 1030.6341504008299*lm12(n,S1,S2) + 30.568594740114495*lm13(n,S1,S2,S3)
    elif variation == 14:
        fit = -3670.6707559192682/np.power(-1. + n,2) + 5507.002671361118/(-1. + n) + 15197.611675043092/np.power(n,3) - 97.9484707672736*lm12(n,S1,S2) + 191.54370342235256*lm13(n,S1,S2,S3)
    elif variation == 15:
        fit = -1722.0358945603757/np.power(-1. + n,2) + 1005.1506697978073/(-1. + n) + 14472.81254861761/np.power(n,2) + 890.1688417705905*lm12(n,S1,S2) + 350.5641184465057*lm13(n,S1,S2,S3)
    elif variation == 16:
        fit = -7677.684552786546/np.power(-1. + n,2) + 8837.926288795223/(-1. + n) + 57514.561003940005/np.power(n,4) + 1800.00821732893*lm11(n,S1) + 163.14762960523643*lm13(n,S1,S2,S3)
    elif variation == 17:
        fit = -3668.0356413358095/np.power(-1. + n,2) + 5640.580814304368/(-1. + n) + 14627.476889260639/np.power(n,3) + 232.17686518512448*lm11(n,S1) + 202.60567395706914*lm13(n,S1,S2,S3)
    elif variation == 18:
        fit = -691.6734321623419/np.power(-1. + n,2) - 3417.4240793927916/(-1. + n) + 22411.6434750231/np.power(n,2) - 3394.848223236357*lm11(n,S1) + 276.04609873762405*lm13(n,S1,S2,S3)
    elif variation == 19:
        fit = -3074.255121846279/np.power(-1. + n,2) + 5167.092611529961/(-1. + n) - 8517.21102404984/np.power(n,4) + 16793.629342505785/np.power(n,3) + 208.44893320267045*lm13(n,S1,S2,S3)
    elif variation == 20:
        fit = -5257.044527918616/np.power(-1. + n,2) + 4591.469890746334/(-1. + n) + 37585.871230195466/np.power(n,4) + 7765.593309542773/np.power(n,2) + 202.2667418438865*lm13(n,S1,S2,S3)
    elif variation == 21:
        fit = -3477.5096784512552/np.power(-1. + n,2) + 5060.750476829951/(-1. + n) + 13691.12777479692/np.power(n,3) + 1434.6372023383353/np.power(n,2) + 207.3068180594*lm13(n,S1,S2,S3)
    else:
        fit = -4659.8548692239365/np.power(-1. + n,2) + 8775.661049395696/(-1. + n) + 11906.360571226945/np.power(n,4) + 4192.865495685177/np.power(n,3) - 1036.6448907424908/np.power(n,2) - 5192.654929396816/(1. + n) - 520.4764518573357/(2. + n) - 302.22022583075585*lm11(n,S1) + 214.3960445752241*lm12(n,S1,S2) + 245.07815446568293*lm13(n,S1,S2,S3)
    return common + fit



@nb.njit(cache=True)
def gamma_gq_nf2(n, cache, variation):
    r"""Return the part proportional to :math:`nf^2` of :math:`\gamma_{gq}^{(3)}`.

    This therm is parametrized using the analytic result from :cite:`Falcioni:2023tzp`
    with an higher number of moments (30).

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^2}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    Lm11 = lm11(n,S1)
    Lm12 = lm12(n,S1, S2)
    Lm13 = lm13(n,S1, S2, S3)
    Lm14 = lm14(n,S1, S2, S3, S4)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    Lm14m1 = lm14m1(n, S1, S2, S3, S4)
    Lm11m2 = lm11m2(n, S1)
    Lm12m2 = lm12m2(n, S1, S2)
    Lm13m2 = lm13m2(n, S1, S2, S3)
    Lm14m2 = lm14m2(n, S1, S2, S3, S4)
    return (
        -(70.60121231446594/(-1. + n)**2)
        - 699.5449657900476/(-1. + n)
        + 617.4606265472538/n**5
        + 21.0418422974213/n**4
        + 656.9409510996688/n**3
        + 440.98264702900605/n**2
        - 485.09325526270226/n
        + 468.97972206118425/(1 + n)**2
        + 131.12265149192916/(1 + n)
        - 284.0960143480868/(2 + n)**2
        + 189.98763175661884/(2 + n)
        + 355.07676818390956/(3 + n)**2
        + 259.2485292950681/(3 + n)
        + 592.4002328363352/(n + n**2)
        + 54.543536161068644/(3 + 4 * n + n**2)
        - 62.424886245567585/(6 + 5 * n + n**2)
        + 154.10095015747495 * (1/n - n/(2 + 3 * n + n**2))
        - 645.1788277783346 * Lm11
        + 32.22330776302828 * Lm11m1
        - 476.25599212133864 * Lm11m2
        - 212.9330738830414 * Lm12
        - 93.58928584449357 * Lm12m1
        + 105.52933047599603 * Lm12m2
        - 26.13260173754001 * Lm13
        - 22.482518440225107 * Lm13m1
        - 45.725204763960996 * Lm13m2
        - 0.877914951989026 * Lm14
        - 0.40377681107870367 * Lm14m1
        + 20.629383319025006 * Lm14m2
    )

@nb.njit(cache=True)
def gamma_gq(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-quark singlet anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| gluon-quark singlet anomalous dimension
        :math:`\gamma_{gq}^{(3)}(N)`

    """
    return (
        gamma_gq_nf0(n, cache, variation)
        + nf * gamma_gq_nf1(n, cache, variation)
        + nf**2 * gamma_gq_nf2(n, cache, variation)
        + nf**3 * gamma_gq_nf3(n, cache)
    )
