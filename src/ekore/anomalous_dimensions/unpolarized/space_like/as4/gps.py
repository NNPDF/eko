# pylint: skip-file
# fmt: off
r"""The anomalous dimension :math:`\gamma_{ps}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics.log_functions import lm11m1, lm12m1, lm13m1, lm14m1


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
    variation : int
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
        fit = 2199.5733178661926/np.power(-1. + n,2) + 50220.7610650693/np.power(n,4) - 25639.474370323645/np.power(n,3) - (30.46463739672361*np.power(S1,3))/np.power(n,2)
    elif variation == 2:
        fit = 7596.245323405076/np.power(-1. + n,2) - 44312.43712535094/np.power(n,4) - 11176.594525482886/np.power(n,2) + (89.08169320197096*np.power(S1,3))/np.power(n,2)
    elif variation == 3:
        fit = -10210.12421512859*(1/(-1. + n) - 1./n) + 8068.074302547925/np.power(-1. + n,2) - 14490.265233715672/np.power(n,4) + (59.68762365947314*np.power(S1,3))/np.power(n,2)
    elif variation == 4:
        fit = 10724.19395051055/np.power(-1. + n,2) - 91906.68748492465/np.power(n,4) - 18075.19853800384/(n + np.power(n,2)) + (166.20937161912028*np.power(S1,3))/np.power(n,2)
    elif variation == 5:
        fit = 15061.364290251815/np.power(-1. + n,2) - 152454.88683661324/np.power(n,4) - 29458.644643354506/(2.*n + np.power(n,2)) + (304.7452062754471*np.power(S1,3))/np.power(n,2)
    elif variation == 6:
        fit = 18207.919823884586/np.power(-1. + n,2) - 198879.253313494/np.power(n,4) - 35873.11939644635/np.power(1. + n,2) + (374.124697728681*np.power(S1,3))/np.power(n,2)
    elif variation == 7:
        fit = 4056.8713008444715/np.power(-1. + n,2) - 1580.9624428578/np.power(n,4) - 48639.30941850732/np.power(1. + n,3) - (57.923784814951865*np.power(S1,3))/np.power(n,2)
    elif variation == 8:
        fit = 15328.653205439949/np.power(-1. + n,2) - 154876.36205635773/np.power(n,4) - (10717.031829005755*S1)/np.power(n,2) + (566.2084850946995*np.power(S1,3))/np.power(n,2)
    elif variation == 9:
        fit = 5414.403316469745/np.power(-1. + n,2) - 19361.30773107909/np.power(n,3) - (2624.2079197404937*S1)/np.power(n,2) + (115.63872018596471*np.power(S1,3))/np.power(n,2)
    elif variation == 10:
        fit = 4497.2070221323/np.power(-1. + n,2) - 15656.01349052962/np.power(n,2) + (4295.232820194323*S1)/np.power(n,2) - (102.14388946214252*np.power(S1,3))/np.power(n,2)
    elif variation == 11:
        fit = -11263.985041057245*(1/(-1. + n) - 1./n) + 7318.657398492708/np.power(-1. + n,2) + (1106.1824299927946*S1)/np.power(n,2) + (7.405939114306042*np.power(S1,3))/np.power(n,2)
    elif variation == 12:
        fit = 4003.806706015412/np.power(-1. + n,2) - 44456.653334559924/(n + np.power(n,2)) + (15641.924494257852*S1)/np.power(n,2) - (417.6048812994036*np.power(S1,3))/np.power(n,2)
    elif variation == 13:
        fit = 5194.538684437648/np.power(-1. + n,2) + 126262.11753376666/np.power(1. + n,2) - (48437.61914266576*S1)/np.power(n,2) + (1242.2831099052773*np.power(S1,3))/np.power(n,2)
    elif variation == 14:
        fit = 3940.6234297395335/np.power(-1. + n,2) - 49140.93517917013/np.power(1. + n,3) + (110.5266359153311*S1)/np.power(n,2) - (64.36057078890452*np.power(S1,3))/np.power(n,2)
    elif variation == 15:
        fit = 3782.941245959016/np.power(-1. + n,2) - (8396.526563401267*S1)/np.power(n,2) - (3187.7284289163363*np.power(S1,3))/np.power(n,2) + 23664.13083030526*lm12m1(n,S1,S2)
    elif variation == 16:
        fit = 4033.1787588235884/np.power(-1. + n,2) - (11884.628852599903*S1)/np.power(n,2) - (270.18257796608805*np.power(S1,3))/np.power(n,2) - 19866.64076207529*lm11m1(n,S1)
    elif variation == 17:
        fit = 4038.8081319842377/np.power(-1. + n,2) - (13092.144791287023*S1)/np.power(n,2) + (6789.885523102248*np.power(S1,2))/np.power(n,2) - (996.5420938163408*np.power(S1,3))/np.power(n,2)
    else:
        fit = 7262.768247576751/np.power(-1. + n,2) - 1263.1828974226962/(-1. + n) - 35781.18196636733/np.power(n,4) - 2647.1048294942784/np.power(n,3) - 1578.388706824265/np.power(n,2) + 1263.1828974226962/n - 5751.779093981027/np.power(1. + n,3) + 5316.999890430606/np.power(1. + n,2) - 1732.8614496090886/(n*(2. + n)) - 3678.3442277978684/(n + np.power(n,2)) - (4352.840748137641*S1)/np.power(n,2) + (399.4050307707205*np.power(S1,2))/np.power(n,2) - (129.50388339270296*np.power(S1,3))/np.power(n,2) - 1168.6259271808995*lm11m1(n,S1) + 1392.0076959003095*lm12m1(n,S1,S2)
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
    variation : int
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
    if variation == 1:
        fit = 82.31518196362028/np.power(-1. + n,2) + 2925.4618402000056/np.power(n,4) - 1096.1635358250617/np.power(n,3) + (13.727225301333212*np.power(S1,3))/np.power(n,2)
    elif variation == 2:
        fit = 313.03890827084507/np.power(-1. + n,2) - 1116.1126197351734/np.power(n,4) - 477.83254822558797/np.power(n,2) + (18.83818532146737*np.power(S1,3))/np.power(n,2)
    elif variation == 3:
        fit = -436.5130774218474*(1/(-1. + n) - 1./n) + 333.2109958344148/np.power(-1. + n,2) + 158.8736344508162/np.power(n,4) + (17.581501711787485*np.power(S1,3))/np.power(n,2)
    elif variation == 4:
        fit = 446.76798467249284/np.power(-1. + n,2) - 3150.9079125544044/np.power(n,4) - 772.7683202075025/(n + np.power(n,2)) + (22.135622213604815*np.power(S1,3))/np.power(n,2)
    elif variation == 5:
        fit = 632.1948739455884/np.power(-1. + n,2) - 5739.522925895252/np.power(n,4) - 1259.4443866700044/(2.*n + np.power(n,2)) + (28.058439825307655*np.power(S1,3))/np.power(n,2)
    elif variation == 6:
        fit = 766.719450760205/np.power(-1. + n,2) - 7724.302214687518/np.power(n,4) - 1533.6821976427764/np.power(1. + n,2) + (31.024618755759825*np.power(S1,3))/np.power(n,2)
    elif variation == 7:
        fit = 161.7201765706361/np.power(-1. + n,2) + 710.7845975180854/np.power(n,4) - 2079.474665595786/np.power(1. + n,3) + (12.553265348412015*np.power(S1,3))/np.power(n,2)
    elif variation == 8:
        fit = 643.6222674298624/np.power(-1. + n,2) - 5843.048169839509/np.power(n,4) - (458.18488060855356*S1)/np.power(n,2) + (39.236770085098755*np.power(S1,3))/np.power(n,2)
    elif variation == 9:
        fit = 269.58559133594/np.power(-1. + n,2) - 730.4475144025814/np.power(n,3) - (152.86546772966202*S1)/np.power(n,2) + (22.23804409954797*np.power(S1,3))/np.power(n,2)
    elif variation == 10:
        fit = 234.9823629423017/np.power(-1. + n,2) - 590.6572168807367/np.power(n,2) + (108.18550877168953*S1)/np.power(n,2) + (14.021720692724948*np.power(S1,3))/np.power(n,2)
    elif variation == 11:
        fit = -424.95837521867156*(1/(-1. + n) - 1./n) + 341.4277251589481/np.power(-1. + n,2) - (12.128364814859474*S1)/np.power(n,2) + (18.154726639345313*np.power(S1,3))/np.power(n,2)
    elif variation == 12:
        fit = 216.3677611024424/np.power(-1. + n,2) - 1677.2240996283547/(n + np.power(n,2)) + (536.2641719039273*S1)/np.power(n,2) + (2.12026751719307*np.power(S1,3))/np.power(n,2)
    elif variation == 13:
        fit = 261.29071909881907/np.power(-1. + n,2) + 4763.513456671153/np.power(1. + n,2) - (1881.2762145084898*S1)/np.power(n,2) + (64.74315819187058*np.power(S1,3))/np.power(n,2)
    elif variation == 14:
        fit = 213.98403435481964/np.power(-1. + n,2) - 1853.9488373207428/np.power(1. + n,3) - (49.691648766849966*S1)/np.power(n,2) + (15.447178703830566*np.power(S1,3))/np.power(n,2)
    elif variation == 15:
        fit = 208.0351304477161/np.power(-1. + n,2) - (370.638764046496*S1)/np.power(n,2) - (102.38867826526625*np.power(S1,3))/np.power(n,2) + 892.7808898852011*lm12m1(n,S1,S2)
    elif variation == 16:
        fit = 217.4758857841338/np.power(-1. + n,2) - (502.2350220325821*S1)/np.power(n,2) + (7.682095082408119*np.power(S1,3))/np.power(n,2) - 749.5123039076917*lm11m1(n,S1)
    elif variation == 17:
        fit = 217.68826615068505/np.power(-1. + n,2) - (547.7911914952281*S1)/np.power(n,2) + (256.1632236993354*np.power(S1,2))/np.power(n,2) - (19.72140008872648*np.power(S1,3))/np.power(n,2)
    else:
        fit = 327.0839597543219/np.power(-1. + n,2) - 50.67479133179523/(-1. + n) - 1163.4572806201736/np.power(n,4) - 107.44770883692019/np.power(n,3) - 62.85233912390145/np.power(n,2) + 50.67479133179523/n - 231.377853112737/np.power(1. + n,3) + 189.99007406049276/np.power(1. + n,2) - 74.08496392176497/(n*(2. + n)) - 144.11720116681514/(n + np.power(n,2)) - (195.90363960747678*S1)/np.power(n,2) + (15.068424923490317*np.power(S1,2))/np.power(n,2) + (12.085455360923469*np.power(S1,3))/np.power(n,2) - 44.08895905339363*lm11m1(n,S1) + 52.51652293442359*lm12m1(n,S1,S2)
    return common + fit


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
    variation : int
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
