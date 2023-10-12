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
        fit = -134611.31548520518/np.power(-1. + n,2) + 105578.6478973615/(-1. + n) - 271004.1457372756/(1. + n) + 135827.42962246042/(2. + n) - 1659.6381024386487*lm13(n,S1,S2,S3)
    elif variation == 2:
        fit = -129784.96381286802/np.power(-1. + n,2) + 92404.11351667382/(-1. + n) - 166871.00811514194/(1. + n) + 8603.419512104943*lm12(n,S1,S2) - 333.06211910306587*lm13(n,S1,S2,S3)
    elif variation == 3:
        fit = -130354.69780717866/np.power(-1. + n,2) + 93906.9798287439/(-1. + n) - 206720.3072485826/(1. + n) - 30775.363703337393*lm11(n,S1) - 1101.8494545818282*lm13(n,S1,S2,S3)
    elif variation == 4:
        fit = -156517.8307292757/np.power(-1. + n,2) + 89433.22729993386/(-1. + n) + 384766.5696200253/np.power(n,4) - 128852.37353471664/(1. + n) - 1762.2085163661038*lm13(n,S1,S2,S3)
    elif variation == 5:
        fit = -129351.9314649959/np.power(-1. + n,2) + 54671.293834508826/(-1. + n) + 148449.9518256113/np.power(n,3) - 88593.14563810707/(1. + n) - 1703.1788616091594*lm13(n,S1,S2,S3)
    elif variation == 6:
        fit = -289524.23824347754/np.power(-1. + n,2) + 762852.3222214412/(-1. + n) - 1.1600176710948618e6/np.power(n,2) - 809182.8193866021/(1. + n) - 1836.245183321374*lm13(n,S1,S2,S3)
    elif variation == 7:
        fit = -122050.84394025133/np.power(-1. + n,2) + 71292.218842658/(-1. + n) - 217660.39733703827/(2. + n) + 22390.205543958295*lm12(n,S1,S2) + 1792.7460216611355*lm13(n,S1,S2,S3)
    elif variation == 8:
        fit = -116666.509065989/np.power(-1. + n,2) + 56373.894282328176/(-1. + n) - 436786.1136556859/(2. + n) - 129741.02583569557*lm11(n,S1) + 691.8555166149675*lm13(n,S1,S2,S3)
    elif variation == 9:
        fit = -176374.82189677586/np.power(-1. + n,2) + 74798.33577898303/(-1. + n) + 733535.2482241682/np.power(n,4) - 123119.72215889738/(2. + n) - 1855.182674086565*lm13(n,S1,S2,S3)
    elif variation == 10:
        fit = -126797.5604335579/np.power(-1. + n,2) + 29946.67399201152/(-1. + n) + 220548.93815270695/np.power(n,3) - 65968.49558222751/(2. + n) - 1724.3256828880108*lm13(n,S1,S2,S3)
    elif variation == 11:
        fit = -56603.682794921304/np.power(-1. + n,2) - 225396.73137215254/(-1. + n) + 584136.1120157596/np.power(n,2) + 204224.41065290783/(2. + n) - 1570.7062041132795*lm13(n,S1,S2,S3)
    elif variation == 12:
        fit = -127399.17314512776/np.power(-1. + n,2) + 86110.78284306003/(-1. + n) + 128873.43260633861*lm11(n,S1) + 44630.68519659024*lm12(n,S1,S2) + 2886.2747447077363*lm13(n,S1,S2,S3)
    elif variation == 13:
        fit = -247120.59412629303/np.power(-1. + n,2) + 79364.32964611659/(-1. + n) + 1.6888135534086183e6/np.power(n,4) - 29158.622788111865*lm12(n,S1,S2) - 6605.8570968113345*lm13(n,S1,S2,S3)
    elif variation == 14:
        fit = -128861.83507443611/np.power(-1. + n,2) + 11966.126835094941/(-1. + n) + 316462.3091625236/np.power(n,3) - 9737.159060072763*lm12(n,S1,S2) - 3253.8465506459015*lm13(n,S1,S2,S3)
    elif variation == 15:
        fit = -88285.0983393474/np.power(-1. + n,2) - 81776.66054398555/(-1. + n) + 301369.70052571065/np.power(n,2) + 10838.566464150015*lm12(n,S1,S2) + 57.46109548925452*lm13(n,S1,S2,S3)
    elif variation == 16:
        fit = -199811.41276401555/np.power(-1. + n,2) + 82030.26173438856/(-1. + n) + 1.021461077715438e6/np.power(n,4) + 50925.69521801989*lm11(n,S1) - 2854.941152739061*lm13(n,S1,S2,S3)
    elif variation == 17:
        fit = -128599.87560093393/np.power(-1. + n,2) + 25245.26855710262/(-1. + n) + 259784.61882981652/np.power(n,3) + 23080.94295557716*lm11(n,S1) - 2154.164587558643*lm13(n,S1,S2,S3)
    elif variation == 18:
        fit = -75739.554722418/np.power(-1. + n,2) - 135625.28803179998/(-1. + n) + 398031.7522689792/np.power(n,2) - 41335.178649992326*lm11(n,S1) - 849.8594840371891*lm13(n,S1,S2,S3)
    elif variation == 19:
        fit = -69571.53835633998/np.power(-1. + n,2) - 21824.684729051125/(-1. + n) - 846704.7809898313/np.power(n,4) + 475124.08069442277/np.power(n,3) - 1573.2801259880794*lm13(n,S1,S2,S3)
    elif variation == 20:
        fit = -131326.8497758187/np.power(-1. + n,2) - 38110.15884910321/(-1. + n) + 457640.10637686535/np.power(n,4) + 219703.5737179606/np.power(n,2) - 1748.1862124636395*lm13(n,S1,S2,S3)
    elif variation == 21:
        fit = -109659.49203855239/np.power(-1. + n,2) - 32396.266625595177/(-1. + n) + 166701.1822848159/np.power(n,3) + 142618.77212791165/np.power(n,2) - 1686.818976852953*lm13(n,S1,S2,S3)
    else:
        fit = -136905.419981799/np.power(-1. + n,2) + 56230.69937898662/(-1. + n) + 163786.27496929924/np.power(n,4) + 75574.81337856653/np.power(n,3) + 23135.3447410219/np.power(n,2) - 79582.08569811552/(1. + n) - 23975.375640880044/(2. + n) + 48.97631385287552*lm11(n,S1) + 2265.099755648517*lm12(n,S1,S2) - 1373.5720765300828*lm13(n,S1,S2,S3)
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
        fit = -4488.023094730909/np.power(-1. + n,2) + 11415.451852682689/(-1. + n) - 22278.11557093013/(1. + n) + 16857.05748831136/(2. + n) + 218.39366882799368*lm13(n,S1,S2,S3)
    elif variation == 2:
        fit = -3889.0418151355752/np.power(-1. + n,2) + 9780.407413909863/(-1. + n) - 9354.523537908572/(1. + n) + 1067.7396878871195*lm12(n,S1,S2) + 383.03027923765*lm13(n,S1,S2,S3)
    elif variation == 3:
        fit = -3959.7494676460906/np.power(-1. + n,2) + 9966.92278779446/(-1. + n) - 14300.077821225603/(1. + n) - 3819.420544236109*lm11(n,S1) + 287.618827026041*lm13(n,S1,S2,S3)
    elif variation == 4:
        fit = -7206.762511830159/np.power(-1. + n,2) + 9411.701329387806/(-1. + n) + 47752.005627238184/np.power(n,4) - 4636.1664981662325/(1. + n) + 205.6640208549657*lm13(n,S1,S2,S3)
    elif variation == 5:
        fit = -3835.299716042018/np.power(-1. + n,2) + 5097.521841045809/(-1. + n) + 18423.593665995402/np.power(n,3) + 360.2625808084624/(1. + n) + 212.9899806152177*lm13(n,S1,S2,S3)
    elif variation == 6:
        fit = -23713.713015219728/np.power(-1. + n,2) + 92987.34159859698/(-1. + n) - 143965.65276583107/np.power(n,2) - 89069.54986300123/(1. + n) + 196.47559437456405*lm13(n,S1,S2,S3)
    elif variation == 7:
        fit = -3455.4793454735536/np.power(-1. + n,2) + 8596.908147688335/(-1. + n) - 12201.695987567426/(2. + n) + 1840.6050040604714*lm12(n,S1,S2) + 502.1997054577737*lm13(n,S1,S2,S3)
    elif variation == 8:
        fit = -3012.8557307127066/np.power(-1. + n,2) + 7370.535179202942/(-1. + n) - 30215.103197366938/(2. + n) - 10665.466242204988*lm11(n,S1) + 411.70010872417294*lm13(n,S1,S2,S3)
    elif variation == 9:
        fit = -7921.226007375836/np.power(-1. + n,2) + 8885.131333721463/(-1. + n) + 60300.859940087874/np.power(n,4) - 4429.9031168628235/(2. + n) + 202.31876873895882*lm13(n,S1,S2,S3)
    elif variation == 10:
        fit = -3845.68702415166/np.power(-1. + n,2) + 5198.064104104413/(-1. + n) + 18130.404314827305/np.power(n,3) + 268.2598106136584/(2. + n) + 213.07597381881035*lm13(n,S1,S2,S3)
    elif variation == 11:
        fit = 1924.6576108019278/np.power(-1. + n,2) - 15792.64760438993/(-1. + n) + 48019.38279297407/np.power(n,2) + 22479.68678039739/(2. + n) + 225.7043876490835*lm13(n,S1,S2,S3)
    elif variation == 12:
        fit = -3755.2981815948488/np.power(-1. + n,2) + 9427.613297165519/(-1. + n) + 7224.439837356762*lm11(n,S1) + 3087.370950732838*lm12(n,S1,S2) + 563.5011879300971*lm13(n,S1,S2,S3)
    elif variation == 13:
        fit = -10466.690790833369/np.power(-1. + n,2) + 9049.417851827156/(-1. + n) + 94672.20408711508/np.power(n,4) - 1049.1403952794576*lm12(n,S1,S2) + 31.387360972561613*lm13(n,S1,S2,S3)
    elif variation == 14:
        fit = -3837.292685015662/np.power(-1. + n,2) + 5271.181705657407/(-1. + n) + 17740.37415701935/np.power(n,3) + 39.59599839782818*lm12(n,S1,S2) + 219.29574546428225*lm13(n,S1,S2,S3)
    elif variation == 15:
        fit = -1562.6253047002033/np.power(-1. + n,2) + 16.11001314684768/(-1. + n) + 16894.306500710383/np.power(n,2) + 1193.038474115512*lm12(n,S1,S2) + 404.9223900617979*lm13(n,S1,S2,S3)
    elif variation == 16:
        fit = -8764.48511413068/np.power(-1. + n,2) + 9145.339290408388/(-1. + n) + 70660.56110839365/np.power(n,4) + 1832.3294758866361*lm11(n,S1) + 166.3470078948553*lm13(n,S1,S2,S3)
    elif variation == 17:
        fit = -3838.357938938717/np.power(-1. + n,2) + 5217.182292574578/(-1. + n) + 17970.85305971273/np.power(n,3) - 93.85827782443143*lm11(n,S1) + 214.8239067092436*lm13(n,S1,S2,S3)
    elif variation == 18:
        fit = -181.6939552271119/np.power(-1. + n,2) - 5911.194541729347/(-1. + n) + 27534.232647597233/np.power(n,2) - 4549.9059886744935*lm11(n,S1) + 305.05047772799065*lm13(n,S1,S2,S3)
    elif variation == 19:
        fit = -4078.395708944439/np.power(-1. + n,2) + 5408.59148938392/(-1. + n) + 3443.111172846758/np.power(n,4) + 17095.178652601353/np.power(n,3) + 212.46174952689813*lm13(n,S1,S2,S3)
    elif variation == 20:
        fit = -6300.379658737184/np.power(-1. + n,2) + 4822.632787569217/(-1. + n) + 50374.02833779294/np.power(n,4) + 7905.033644756689/np.power(n,2) + 206.16854967191063*lm13(n,S1,S2,S3)
    elif variation == 21:
        fit = -3915.3786995698256/np.power(-1. + n,2) + 5451.580654627375/(-1. + n) + 18349.375335216806/np.power(n,3) - 579.9569091789359/np.power(n,2) + 212.92345341087133*lm13(n,S1,S2,S3)
    else:
        fit = -5243.037055009921/np.power(-1. + n,2) + 9562.656801160758/(-1. + n) + 15581.084298736878/np.power(n,4) + 5129.037104065377/np.power(n,3) - 2104.412099474839/np.power(n,2) - 6632.29384335349/(1. + n) - 344.8427724987991/(2. + n) - 479.6134161760297*lm11(n,S1) + 294.2480819006815*lm12(n,S1,S2) + 266.47872117598763*lm13(n,S1,S2,S3)
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
