# -*- coding: utf-8 -*-
# pylint: skip-file
# fmt: off
r"""The anomalous dimension :math:`\gamma_{ps}^{(3)}`."""
import numba as nb
import numpy as np

from ...harmonics.log_functions import lm11m1, lm12m1, lm13m1, lm14m1


@nb.njit(cache=True)
def gamma_ps_nf3(n, sx):
    r"""Implement the part proportional to :math:`nf^3` of :math:`\gamma_{ps}^{(3)}`.

    The expression is copied exact from Eq. 3.10 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    return 1.3333333333333333 * (
        16.305796943701882 / (-1.0 + n)
        + 3.5555555555555554 / np.power(n, 5)
        - 17.185185185185187 / np.power(n, 4)
        + 28.839506172839506 / np.power(n, 3)
        - 48.95252257604665 / np.power(n, 2)
        + 23.09354523864061 / n
        + 39.111111111111114 / np.power(1.0 + n, 5)
        - 61.03703703703704 / np.power(1.0 + n, 4)
        - 10.666666666666666 / np.power(1.0 + n, 3)
        + 59.29439100420026 / np.power(1.0 + n, 2)
        - 94.20465634975173 / (1.0 + n)
        + 18.962962962962962 / np.power(2.0 + n, 4)
        - 34.76543209876543 / np.power(2.0 + n, 3)
        + 14.222222222222221 / np.power(2.0 + n, 2)
        + 54.805314167409236 / (2.0 + n)
        - (1.5802469135802468 * S1) / (-1.0 + n)
        + (7.111111111111111 * S1) / np.power(n, 4)
        - (34.370370370370374 * S1) / np.power(n, 3)
        + (38.71604938271605 * S1) / np.power(n, 2)
        - (37.135802469135804 * S1) / n
        + (35.55555555555556 * S1) / np.power(1.0 + n, 4)
        - (43.851851851851855 * S1) / np.power(1.0 + n, 3)
        - (39.50617283950617 * S1) / np.power(1.0 + n, 2)
        + (89.28395061728395 * S1) / (1.0 + n)
        + (18.962962962962962 * S1) / np.power(2.0 + n, 3)
        - (34.76543209876543 * S1) / np.power(2.0 + n, 2)
        - (50.5679012345679 * S1) / (2.0 + n)
        + (4.7407407407407405 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        + (7.111111111111111 * (np.power(S1, 2) + S2)) / np.power(n, 3)
        - (15.407407407407407 * (np.power(S1, 2) + S2)) / np.power(n, 2)
        + (13.037037037037036 * (np.power(S1, 2) + S2)) / n
        + (14.222222222222221 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 3)
        - (4.7407407407407405 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        - (20.14814814814815 * (np.power(S1, 2) + S2)) / (1.0 + n)
        + (9.481481481481481 * (np.power(S1, 2) + S2)) / np.power(2.0 + n, 2)
        + (2.3703703703703702 * (np.power(S1, 2) + S2)) / (2.0 + n)
        - (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        + (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(n, 2)
        - (1.1851851851851851 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        + (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(1.0 + n, 2)
        + (1.1851851851851851 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
        + (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (2.0 + n)
    )


@nb.njit(cache=True)
def gamma_ps_nf1(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^1` of :math:`\gamma_{ps}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : str
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    common = -3498.454512979491/np.power(-1. + n,3) + 5404.444444444444/np.power(n,7) + 3425.9753086419755/np.power(n,6) + 20515.223982421852/np.power(n,5) + 247.55054124312667*lm13m1(n,S1,S2,S3) + 56.46090534979424*lm14m1(n,S1,S2,S3,S4)
    if variation == 1:
        fit = 6056.204267761437/np.power(-1. + n,2) + 30316.627474493205/np.power(n,4) - (3513.605730075393*n)/np.power(-1. + n,2) + (1403.2000303082746*np.power(S1,3))/np.power(n,2)
    elif variation == 2:
        fit = 2199.5733178661926/np.power(-1. + n,2) + 50220.7610650693/np.power(n,4) - 25639.474370323645/np.power(n,3) - (30.46463739672361*np.power(S1,3))/np.power(n,2)
    elif variation == 3:
        fit = 7596.245323405077/np.power(-1. + n,2) - 44312.43712535094/np.power(n,4) - 11176.594525482886/np.power(n,2) + (89.08169320197096*np.power(S1,3))/np.power(n,2)
    elif variation == 4:
        fit = -10210.124215128611*(1/(-1. + n) - 1./n) + 8068.074302547951/np.power(-1. + n,2) - 14490.265233715903/np.power(n,4) + (59.68762365947316*np.power(S1,3))/np.power(n,2)
    elif variation == 5:
        fit = -5024.075626622562/np.power(-1. + n,2) + 115212.96831222711/np.power(n,4) - 7613.740540053544/(1. + n) + (2926.1163505761606*np.power(S1,3))/np.power(n,2)
    elif variation == 6:
        fit = 171737.67330653904/np.power(n,4) - 26758.396554806663/(2. - 3.*n + np.power(n,3)) + (7669.156759254967*n)/(2. - 3.*n + np.power(n,3)) - (8419.34201466332*np.power(n,2))/(2. - 3.*n + np.power(n,3)) + (3138.9513984212067*np.power(S1,3))/np.power(n,2)
    elif variation == 7:
        fit = 18207.919823884586/np.power(-1. + n,2) - 198879.253313494/np.power(n,4) - 35873.11939644635/np.power(1. + n,2) + (374.124697728681*np.power(S1,3))/np.power(n,2)
    elif variation == 8:
        fit = 4056.871300844477/np.power(-1. + n,2) - 1580.9624428578652/np.power(n,4) - 48639.30941850736/np.power(1. + n,3) - (57.92378481495256*np.power(S1,3))/np.power(n,2)
    elif variation == 9:
        fit = 15328.653205439909/np.power(-1. + n,2) - 154876.36205635726/np.power(n,4) - (10717.031829005737*S1)/np.power(n,2) + (566.208485094698*np.power(S1,3))/np.power(n,2)
    elif variation == 10:
        fit = 7574.130950256209/np.power(-1. + n,2) - (2938.4183200076977*n)/np.power(-1. + n,2) - (1754.4090757178858*S1)/np.power(n,2) + (1266.1820959910863*np.power(S1,3))/np.power(n,2)
    elif variation == 11:
        fit = 5414.403316469749/np.power(-1. + n,2) - 19361.30773107913/np.power(n,3) - (2624.2079197404937*S1)/np.power(n,2) + (115.63872018596449*np.power(S1,3))/np.power(n,2)
    elif variation == 12:
        fit = 4497.2070221323/np.power(-1. + n,2) - 15656.01349052962/np.power(n,2) + (4295.232820194323*S1)/np.power(n,2) - (102.14388946214252*np.power(S1,3))/np.power(n,2)
    elif variation == 13:
        fit = -11263.98504105731*(1/(-1. + n) - 1./n) + 7318.657398492729/np.power(-1. + n,2) + (1106.1824299928326*S1)/np.power(n,2) + (7.405939114302806*np.power(S1,3))/np.power(n,2)
    elif variation == 14:
        fit = 3657.8604528061715/np.power(-1. + n,2) - 4365.920100861775/(1. + n) - (4571.602465122715*S1)/np.power(n,2) + (1919.4420411747392*np.power(S1,3))/np.power(n,2)
    elif variation == 15:
        fit = 3431.485602677905/(2. - 3.*n + np.power(n,3)) + (11696.614112414916*n)/(2. - 3.*n + np.power(n,3)) - (3992.3485244303856*np.power(n,2))/(2. - 3.*n + np.power(n,3)) - (5635.147029185683*S1)/np.power(n,2) + (1786.1715601033065*np.power(S1,3))/np.power(n,2)
    elif variation == 16:
        fit = 5194.538684437648/np.power(-1. + n,2) + 126262.11753376666/np.power(1. + n,2) - (48437.61914266576*S1)/np.power(n,2) + (1242.2831099052773*np.power(S1,3))/np.power(n,2)
    elif variation == 17:
        fit = 3940.6234297395335/np.power(-1. + n,2) - 49140.93517917013/np.power(1. + n,3) + (110.5266359153311*S1)/np.power(n,2) - (64.36057078890452*np.power(S1,3))/np.power(n,2)
    elif variation == 18:
        fit = 3782.941245959017/np.power(-1. + n,2) - (8396.526563401267*S1)/np.power(n,2) - (3187.7284289163345*np.power(S1,3))/np.power(n,2) + 23664.13083030524*lm12m1(n,S1,S2)
    elif variation == 19:
        fit = 4033.17875882359/np.power(-1. + n,2) - (11884.628852599935*S1)/np.power(n,2) - (270.1825779660905*np.power(S1,3))/np.power(n,2) - 19866.640762075374*lm11m1(n,S1)
    else:
        fit = 5363.3161670654745/np.power(-1. + n,2) - 1130.2162766413642/(-1. + n) - 2455.3289480761696/np.power(n,4) - 2368.4622158633038/np.power(n,3) - 1412.2425271585528/np.power(n,2) + 1130.2162766413642/n - (339.5802131622679*n)/np.power(-1. + n,2) - 5146.328663035657/np.power(1. + n,3) + 4757.315691437911/np.power(1. + n,2) - 630.5084547850166/(1. + n) - 1227.7321553751976/(2. - 3.*n + np.power(n,3)) + (1019.2510985089411*n)/(2. - 3.*n + np.power(n,3)) - (653.2468704786161*np.power(n,2))/(2. - 3.*n + np.power(n,3)) - (4658.380578491421*S1)/np.power(n,2) + (588.509992427368*np.power(S1,3))/np.power(n,2) - 1045.6126716881774*lm11m1(n,S1) + 1245.4805700160653*lm12m1(n,S1,S2)
    return common + fit


@nb.njit(cache=True)
def gamma_ps_nf2(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^2` of :math:`\gamma_{ps}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : str
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    common = -568.8888888888889/np.power(n,7) + 455.1111111111111/np.power(n,6) - 1856.79012345679/np.power(n,5) - 40.559670781893004*lm13m1(n,S1,S2,S3) - 18.106995884773664*lm14m1(n,S1,S2,S3,S4)
    if variation == 0:
        fit = 247.19758779416804/np.power(-1. + n,2) + 2074.501124740322/np.power(n,4) - (150.21706080810173*n)/np.power(-1. + n,2) + (75.02063984800382*np.power(S1,3))/np.power(n,2)
    elif variation == 2:
        fit = 82.31518196362028/np.power(-1. + n,2) + 2925.4618402000056/np.power(n,4) - 1096.1635358250617/np.power(n,3) + (13.727225301333212*np.power(S1,3))/np.power(n,2)
    elif variation == 3:
        fit = 313.03890827084507/np.power(-1. + n,2) - 1116.1126197351734/np.power(n,4) - 477.83254822558797/np.power(n,2) + (18.83818532146735*np.power(S1,3))/np.power(n,2)
    elif variation == 4:
        fit = -436.51307742184764*(1/(-1. + n) - 1./n) + 333.2109958344149/np.power(-1. + n,2) + 158.87363445081542/np.power(n,4) + (17.581501711787457*np.power(S1,3))/np.power(n,2)
    elif variation == 5:
        fit = -226.5172244337121/np.power(-1. + n,2) + 5704.071365034062/np.power(n,4) - 325.50997850797233/(1. + n) + (140.12982712224627*np.power(S1,3))/np.power(n,2)
    elif variation == 6:
        fit = 8120.670067219227/np.power(n,4) - 1167.446971905148/(2. - 3.*n + np.power(n,3)) + (316.15608829446927*n)/(2. - 3.*n + np.power(n,3)) - (359.95182969881733*np.power(n,2))/(2. - 3.*n + np.power(n,3)) + (149.22915635277718*np.power(S1,3))/np.power(n,2)
    elif variation == 7:
        fit = 766.719450760205/np.power(-1. + n,2) - 7724.302214687518/np.power(n,4) - 1533.6821976427764/np.power(1. + n,2) + (31.024618755759825*np.power(S1,3))/np.power(n,2)
    elif variation == 8:
        fit = 161.72017657063654/np.power(-1. + n,2) + 710.7845975180813/np.power(n,4) - 2079.4746655957883/np.power(1. + n,3) + (12.553265348411983*np.power(S1,3))/np.power(n,2)
    elif variation == 9:
        fit = 643.6222674298614/np.power(-1. + n,2) - 5843.048169839496/np.power(n,4) - (458.1848806085531*S1)/np.power(n,2) + (39.23677008509881*np.power(S1,3))/np.power(n,2)
    elif variation == 10:
        fit = 351.06602146998284/np.power(-1. + n,2) - (110.85823271530677*n)/np.power(-1. + n,2) - (120.0504114085077*S1)/np.power(n,2) + (65.64479950792175*np.power(S1,3))/np.power(n,2)
    elif variation == 11:
        fit = 269.5855913359402/np.power(-1. + n,2) - 730.4475144025828/np.power(n,3) - (152.86546772966196*S1)/np.power(n,2) + (22.238044099547945*np.power(S1,3))/np.power(n,2)
    elif variation == 12:
        fit = 234.9823629423017/np.power(-1. + n,2) - 590.6572168807367/np.power(n,2) + (108.18550877168953*S1)/np.power(n,2) + (14.021720692724948*np.power(S1,3))/np.power(n,2)
    elif variation == 13:
        fit = -424.9583752186764*(1/(-1. + n) - 1./n) + 341.4277251589498/np.power(-1. + n,2) - (12.1283648148571*S1)/np.power(n,2) + (18.154726639345217*np.power(S1,3))/np.power(n,2)
    elif variation == 14:
        fit = 203.3161850269303/np.power(-1. + n,2) - 164.71384733147832/(1. + n) - (226.3351695180497*S1)/np.power(n,2) + (90.29045412205691*np.power(S1,3))/np.power(n,2)
    elif variation == 15:
        fit = 260.091131115292/(2. - 3.*n + np.power(n,3)) + (506.5956799409138*n)/(2. - 3.*n + np.power(n,3)) - (150.62004575330712*np.power(n,2))/(2. - 3.*n + np.power(n,3)) - (266.4597052191758*S1)/np.power(n,2) + (85.26253488261446*np.power(S1,3))/np.power(n,2)
    elif variation == 16:
        fit = 261.29071909881907/np.power(-1. + n,2) + 4763.513456671153/np.power(1. + n,2) - (1881.2762145084898*S1)/np.power(n,2) + (64.74315819187058*np.power(S1,3))/np.power(n,2)
    elif variation == 17:
        fit = 213.98403435481964/np.power(-1. + n,2) - 1853.9488373207428/np.power(1. + n,3) - (49.691648766849966*S1)/np.power(n,2) + (15.447178703830566*np.power(S1,3))/np.power(n,2)
    elif variation == 18:
        fit = 208.0351304477161/np.power(-1. + n,2) - (370.6387640464961*S1)/np.power(n,2) - (102.38867826526625*np.power(S1,3))/np.power(n,2) + 892.780889885201*lm12m1(n,S1,S2)
    elif variation == 19:
        fit = 217.4758857841339/np.power(-1. + n,2) - (502.2350220325834*S1)/np.power(n,2) + (7.682095082407954*np.power(S1,3))/np.power(n,2) - 749.5123039076951*lm11m1(n,S1)
    else:
        fit = 243.28794735840168/np.power(-1. + n,2) - 45.34060277055389/(-1. + n) + 263.73155920528035/np.power(n,4) - 96.1374236961918/np.power(n,3) - 56.23630342664867/np.power(n,2) + 45.34060277055389/n - (13.740804922284658*n)/np.power(-1. + n,2) - 207.02228962718584/np.power(1. + n,3) + 169.99111889623035/np.power(1. + n,2) - 25.801253991550034/(1. + n) - 47.75557056788716/(2. - 3.*n + np.power(n,3)) + (43.30272464396752*n)/(2. - 3.*n + np.power(n,3)) - (26.872203971164442*np.power(n,2))/(2. - 3.*n + np.power(n,3)) - (206.9305336779755*S1)/np.power(n,2) + (40.97038018441789*np.power(S1,3))/np.power(n,2) - 39.448015995141844*lm11m1(n,S1) + 46.988467888694785*lm12m1(n,S1,S2)
    return fit + common


@nb.njit(cache=True)
def gamma_ps(n, nf, sx, variation):
    r"""Compute the |N3LO| pure singlet quark-quark anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : list
        harmonic sums cache
    variation : str
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| pure singlet quark-quark anomalous dimension
        :math:`\gamma_{ps}^{(3)}(N)`

    See Also
    --------
    gamma_ps_nf1: :math:`\gamma_{ps}^{(3)}|_{nf^1}`
    gamma_ps_nf2: :math:`\gamma_{ps}^{(3)}|_{nf^2}`
    gamma_ps_nf3: :math:`\gamma_{ps}^{(3)}|_{nf^3}`

    """
    return (
        +nf * gamma_ps_nf1(n, sx, variation)
        + nf**2 * gamma_ps_nf2(n, sx, variation)
        + nf**3 * gamma_ps_nf3(n, sx)
    )
