# pylint: skip-file
# fmt: off
r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{ps}^{(3)}`."""
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .....harmonics.log_functions import (
    lm11m1,
    lm11m2,
    lm12m1,
    lm12m2,
    lm13m1,
    lm13m2,
    lm14m1,
    lm14m2,
)


@nb.njit(cache=True)
def gamma_ps_nf3(n, cache):
    r"""Return the part proportional to :math:`nf^3` of :math:`\gamma_{ps}^{(3)}`.

    The expression is copied exact from :eqref:`3.10` of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^3}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
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
def gamma_ps_nf1(n, cache, variation):
    r"""Return the part proportional to :math:`nf^1` of :math:`\gamma_{ps}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^1}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    common = -3498.454512979491/np.power(-1. + n,3) + 5404.444444444444/np.power(n,7) + 3425.9753086419755/np.power(n,6) + 20515.223982421852/np.power(n,5) + 247.55054124312667*lm13m1(n,S1,S2,S3) + 199.11111*lm13m2(n,S1,S2,S3) + 56.46090534979424*lm14m1(n,S1,S2,S3,S4) + 13.168724000000001*lm14m2(n,S1,S2,S3,S4)
    if variation == 1:
        fit = 17114.440372984987*(1/(n-1)**2 +1/n**2) - 56482.37473713895*(1/(-1. + n) - 1./n) - 9247.158633468189/np.power(n,4) + 51758.36314422693/np.power(n,3) - 27774.710120695992/(6. + 5.*n + np.power(n,2)) + 12609.163387951978*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 218.72209329346168*lm11m1(n,S1) + 6520.842881982454*lm11m2(n,S1) + 1003.6502844579335*lm12m1(n,S1,S2) + 8612.716120193661*lm12m2(n,S1,S2)
    elif variation == 2:
        fit = 17599.147585763596*(1/(n-1)**2 +1/n**2) - 59779.7559042966*(1/(-1. + n) - 1./n) - 4089.771154118181/np.power(n,4) + 52480.70138123076/np.power(n,3) - 40485.48923082326/(3. + 4.*n + np.power(n,2)) + 18807.609859269352*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) + 836.442067628552*lm11m1(n,S1) + 4387.337147765067*lm11m2(n,S1) + 1087.2882129831726*lm12m1(n,S1,S2) + 6334.7980934240695*lm12m2(n,S1,S2)
    elif variation == 3:
        fit = 18901.8357366727*(1/(n-1)**2 +1/n**2) - 68641.72300129247*(1/(-1. + n) - 1./n) + 9771.10692324915/np.power(n,4) + 54422.04117180634/np.power(n,3) + 37323.34376598521/(n + np.power(n,2)) - 1856.9289869250597*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) + 3672.279039077716*lm11m1(n,S1) - 1346.624839240501*lm11m2(n,S1) + 1312.0717554826813*lm12m1(n,S1,S2) + 212.71457252750494*lm12m2(n,S1,S2)
    elif variation == 4:
        fit = 21675.67037317274*(1/(n-1)**2 +1/n**2) - 88470.14327682114*(1/(-1. + n) - 1./n) + 61991.75926834194/np.power(n,4) + 9856.528666259726/np.power(n,3) - 64420.64245382356*(-1./np.power(n,2) + np.power(1. + n,-2)) + 16939.98535914983*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) + 5120.79084021425*lm11m1(n,S1) - 2503.219631267419*lm11m2(n,S1) + 1429.3089002730355*lm12m1(n,S1,S2) - 2455.0221677707123*lm12m2(n,S1,S2)
    elif variation == 5:
        fit = 16128.426707522673*(1/(n-1)**2 +1/n**2) - 49774.69065596102*(1/(-1. + n) - 1./n) - 19738.552313464068/np.power(n,4) + 50288.94897400035/np.power(n,3) + 82357.4319998272/(3. + 4.*n + np.power(n,2)) - 84275.29283548871/(6. + 5.*n + np.power(n,2)) - 2365.184948828527*lm11m1(n,S1) + 10860.917150439236*lm11m2(n,S1) + 833.5102113804192*lm12m1(n,S1,S2) + 13246.560397831827*lm12m2(n,S1,S2)
    elif variation == 6:
        fit = 18672.39801083196*(1/(n-1)**2 +1/n**2) - 67080.89700826164*(1/(-1. + n) - 1./n) + 7329.841282147562/np.power(n,4) + 54080.11971524955/np.power(n,3) + 32532.36067022365/(n + np.power(n,2)) - 3565.2781149891907/(6. + 5.*n + np.power(n,2)) + 3172.8140776568566*lm11m1(n,S1) - 336.72357274822184*lm11m2(n,S1) + 1272.4815041110023*lm12m1(n,S1,S2) + 1290.9736265744925*lm12m2(n,S1,S2)
    elif variation == 7:
        fit = 18784.77446150199*(1/(n-1)**2 +1/n**2) - 67845.37521815149*(1/(-1. + n) - 1./n) + 8525.552885161762/np.power(n,4) + 54247.58748616515/np.power(n,3) + 33969.43677178686/(n + np.power(n,2)) - 3638.0412017774465/(3. + 4.*n + np.power(n,2)) + 3417.4514620587406*lm11m1(n,S1) - 831.3676509530529*lm11m2(n,S1) + 1291.8727745454964*lm12m1(n,S1,S2) + 762.8448669655548*lm12m2(n,S1,S2)
    elif variation == 8:
        fit = 19175.859048974005*(1/(n-1)**2 +1/n**2) - 70600.54578256015*(1/(-1. + n) - 1./n) + 14929.920529938649/np.power(n,4) + 50019.462205806536/np.power(n,3) + 33636.20344856993/(n + np.power(n,2)) - 6364.042487074435*(-1./np.power(n,2) + np.power(1. + n,-2)) + 3815.376801261772*lm11m1(n,S1) - 1460.882951467289*lm11m2(n,S1) + 1323.653558206673*lm12m1(n,S1,S2) - 50.8293856367635*lm12m2(n,S1,S2)
    else:
        fit = 18506.56903717808/np.power(-1. + n,2) - 66084.43819806044/(-1. + n) + 8684.087348473578/np.power(n,4) + 47144.21909309317/np.power(n,3) + 27354.654654790327/np.power(n,2) + 71896.9169004912/n - 8848.085617612249/np.power(1. + n,2) + 17182.668082070708/(n + np.power(n,2)) - (5812.4787024307625*n)/(2. + 3.*n + np.power(n,2)) + 4779.237695903312/(3. + 4.*n + np.power(n,2)) - 14451.910133896738/(6. + 5.*n + np.power(n,2)) + 2181.4059057219874*lm11m1(n,S1) + 1911.2848168137841*lm11m2(n,S1) + 1194.2296501800517*lm12m1(n,S1,S2) + 3494.344515513704*lm12m2(n,S1,S2)
    return common + fit


@nb.njit(cache=True)
def gamma_ps_nf2(n, cache, variation):
    r"""Return the part proportional to :math:`nf^2` of :math:`\gamma_{ps}^{(3)}`.

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
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ps}^{(3)}|_{nf^2}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    common = -568.8888888888889/np.power(n,7) + 455.1111111111111/np.power(n,6) - 1856.79012345679/np.power(n,5) - 40.559670781893004*lm13m1(n,S1,S2,S3) - 13.695473*lm13m2(n,S1,S2,S3) - 3.6213991769547325*lm14m1(n,S1,S2,S3,S4)
    if variation == 1:
        fit = 129.36670007880312*(1/(n-1)**2 +1/n**2) + 281.1298772185047*(1/(-1. + n) - 1./n) + 1461.3249956305358/np.power(n,4) - 1416.6957822207867/np.power(n,3) + 1032.901485734173/(6. + 5.*n + np.power(n,2)) - 538.5144723158696*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 540.9977294644839*lm11m1(n,S1) - 301.9365856452302*lm11m2(n,S1) - 206.40218795098986*lm12m1(n,S1,S2) - 293.5852282113812*lm12m2(n,S1,S2)
    elif variation == 2:
        fit = 111.3411404768573*(1/(n-1)**2 +1/n**2) + 403.7547110249561*(1/(-1. + n) - 1./n) + 1269.5292060536797/np.power(n,4) - 1443.5584823153147/np.power(n,3) + 1505.5969139509264/(3. + 4.*n + np.power(n,2)) - 769.0257480447024*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 580.2377934849065*lm11m1(n,S1) - 222.59457235631194*lm11m2(n,S1) - 209.51256443283995*lm12m1(n,S1,S2) - 208.87269134528498*lm12m2(n,S1,S2)
    elif variation == 3:
        fit = 62.896064518487776*(1/(n-1)**2 +1/n**2) + 733.3183706991717*(1/(-1. + n) - 1./n) + 754.063287520183/np.power(n,4) - 1515.7540492064643/np.power(n,3) - 1388.0010726143498/(n + np.power(n,2)) - 0.5414398156502588*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 685.6984968270021*lm11m1(n,S1) - 9.35681200149867*lm11m2(n,S1) - 217.87194157425705*lm12m1(n,S1,S2) + 18.798786229022753*lm12m2(n,S1,S2)
    elif variation == 4:
        fit = -40.25888630988295*(1/(n-1)**2 +1/n**2) + 1470.7090012125375*(1/(-1. + n) - 1./n) - 1187.9479240380433/np.power(n,4) + 141.57317189169493/np.power(n,3) + 2395.71105319829*(-1./np.power(n,2) + np.power(1. + n,-2)) - 699.571519322418*(1/n - (1.*n)/(2. + 3.*n + np.power(n,2))) - 739.5665270870636*lm11m1(n,S1) + 33.65531124702984*lm11m2(n,S1) - 222.23181895849254*lm12m1(n,S1,S2) + 118.00802383230622*lm12m2(n,S1,S2)
    elif variation == 5:
        fit = 171.4775540183464*(1/(n-1)**2 +1/n**2) - 5.343134285176699*(1/(-1. + n) - 1./n) + 1909.3933712709888/np.power(n,4) - 1353.9397651287413/np.power(n,3) - 3517.3363721413493/(3. + 4.*n + np.power(n,2)) + 3445.9387339426257/(6. + 5.*n + np.power(n,2)) - 449.32621253066674*lm11m1(n,S1) - 487.29327172119923*lm11m2(n,S1) - 199.13581554176443*lm12m1(n,S1,S2) - 491.4882800819326*lm12m2(n,S1,S2)
    elif variation == 6:
        fit = 62.82915588751939*(1/(n-1)**2 +1/n**2) + 733.773533781873*(1/(-1. + n) - 1./n) + 753.3514170083771/np.power(n,4) - 1515.853803964272/np.power(n,3) - 1389.3980632773396/(n + np.power(n,2)) - 1.0393493977265766/(6. + 5.*n + np.power(n,2)) - 685.8440673314171*lm11m1(n,S1) - 9.062382940371117*lm11m2(n,S1) - 217.88348007889562*lm12m1(n,S1,S2) + 19.11307505300711*lm12m2(n,S1,S2)
    elif variation == 7:
        fit = 62.861933458087776*(1/(n-1)**2 +1/n**2) + 733.5505587733345*(1/(-1. + n) - 1./n) + 753.7001229299171/np.power(n,4) - 1515.8049105533028/np.power(n,3) - 1388.978986086796/(n + np.power(n,2)) - 1.0607997206025817/(3. + 4.*n + np.power(n,2)) - 685.7728054752505*lm11m1(n,S1) - 9.206574360380268*lm11m2(n,S1) - 217.87783169982788*lm12m1(n,S1,S2) + 18.959202169089114*lm12m2(n,S1,S2)
    elif variation == 8:
        fit = 62.97594965761441*(1/(n-1)**2 +1/n**2) + 732.7473163403877*(1/(-1. + n) - 1./n) + 755.5673108986445/np.power(n,4) - 1517.0376868275046/np.power(n,3) - 1389.0762039654194/(n + np.power(n,2)) - 1.8554880157769447*(-1./np.power(n,2) + np.power(1. + n,-2)) - 685.6567645579319*lm11m1(n,S1) - 9.39011824068036*lm11m2(n,S1) - 217.86856387977127*lm12m1(n,S1,S2) + 18.72193271032573*lm12m2(n,S1,S2)
    else:
        fit = 77.93620147322915/np.power(-1. + n,2) + 635.4550293456986/(-1. + n) + 808.6227234092854/np.power(n,4) - 1267.1339135405867/np.power(n,3) - 221.29574417458494/np.power(n,2) - 886.4116767830286/n + 299.2319456478141/np.power(1. + n,2) - 694.4317907429881/(n + np.power(n,2)) + (250.95664743733002*n)/(2. + 3.*n + np.power(n,2)) - 251.60003223887819/(3. + 4.*n + np.power(n,2)) + 559.725108784884/(6. + 5.*n + np.power(n,2)) - 631.6375495948403*lm11m1(n,S1) - 126.89812575233026*lm11m2(n,S1) - 213.59802551460484*lm12m1(n,S1,S2) - 100.04314745560596*lm12m2(n,S1,S2)
    return common + fit


@nb.njit(cache=True)
def gamma_ps(n, nf, cache, variation):
    r"""Compute the |N3LO| pure singlet quark-quark anomalous dimension.

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
        |N3LO| pure singlet quark-quark anomalous dimension
        :math:`\gamma_{ps}^{(3)}(N)`

    """
    return (
        +nf * gamma_ps_nf1(n, cache, variation)
        + nf**2 * gamma_ps_nf2(n, cache, variation)
        + nf**3 * gamma_ps_nf3(n, cache)
    )
