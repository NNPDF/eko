import numba as nb
import numpy as np

# from eko.constants import zeta2
from numpy import power as npp

from eko import constants

from ....harmonics import w1, w2, w3
from ....harmonics.constants import zeta2 as ZETA2
from ....harmonics.constants import zeta3 as ZETA3
from ....harmonics.polygamma import cern_polygamma as polygamma

# NS = N * N
# NT = NS * N
# NFO = NT * N
# NFI = NFO * N
# NSI = NFI * N
# NSE = NSI * N
# NE = NSE * N
# NN = NE * N

# NM2 = N - 2
# NM = N - 1
# N1 = N + 1
# N2 = N + 2
# NMS = NM * NM
# NMT = NMS * NM
# N1S = N1 * N1
# N1T = N1S * N1
# N2S = N2 * N2
# N2T = N2S * N2

# S1 = w1.S1(N)
# S2 = w2.S2(N)

# N3 = N + 3
# N4 = N + 4
# N5 = N + 5
# N6 = N + 6

# S11 = S1  + 1/N1
# S12 = S11 + 1/N2
# S13 = S12 + 1/N3
# S14 = S13 + 1/N4
# S15 = S14 + 1/N5
# S16 = S15 + 1/N6
# S1M = S1 - 1 / N
# S21 = S2  + 1/N1S
# S22 = S21 + 1/N2S
# S2M = S2 - 1 / NS

# SPMOM = (1.0000 * (ZETA2 - S1 / N ) / N  -
#           0.9992 * (ZETA2 - S11/ N1) / N1 +
#           0.9851 * (ZETA2 - S12/ N2) / N2 -
#           0.9005 * (ZETA2 - S13/ N3) / N3 +
#           0.6621 * (ZETA2 - S14/ N4) / N4 -
#           0.3174 * (ZETA2 - S15/ N5) / N5 +
#           0.0699 * (ZETA2 - S16/ N6) / N6  )

# SLC = - 5/8 * ZETA3
# SLV = (- ZETA2/2* (polygamma(N1/2) - polygamma(N/2))
#           + S1/NS + SPMOM)
# SSCHLM = SLC - SLV
# SSTR2M = ZETA2 - polygamma(N1/2,1)
# SSTR3M = 0.5 * polygamma(N1/2,2) + ZETA3

# SSCHLP = SLC + SLV
# SSTR2P = ZETA2 - polygamma(N2/2,1)
# SSTR3P = 0.5 * polygamma(N2/2,2) + ZETA3

# DS2NM = - polygamma(NM2/2+1,1) + polygamma(NM/2+1,1)
# DS2N  = - polygamma(NM/2+1,1) + polygamma(N/2+1,1)
# DS2N1 = - polygamma(N/2+1,1)  + polygamma(N1/2+1,1)
# DS2N2 = - polygamma(N1/2+1,1) + polygamma(N2/2+1,1)

# PNMA = ( 16* S1 * (2* N + 1) / (NS * N1S) +
#           16* (2* S1 - 1/(N * N1)) * ( S2 - SSTR2M ) +
#           64* SSCHLM + 24* S2 - 3 - 8* SSTR3M -
#           8* (3* NT + NS -1) / (NT * N1T) +
#           16* (2* NS + 2* N +1) / (NT * N1T) ) * (-0.5)
# PNPA = ( 16* S1 * (2* N + 1) / (NS * N1S) +
#           16* (2* S1 - 1/(N * N1)) * ( S2 - SSTR2P ) +
#           64* SSCHLP + 24* S2 - 3 - 8* SSTR3P -
#           8* (3* NT + NS -1) / (NT * N1T) -
#           16* (2* NS + 2* N +1)/(NT * N1T) ) * (-0.5)

# PNSB = ( S1 * (536/9 + 8* (2* N + 1) / (NS * N1S)) -
#           (16* S1 + 52/3- 8/(N * N1)) * S2 - 43/6 -
#           (151* NFO + 263* NT + 97* NS + 3* N + 9) *
#           4/ (9* NT * N1T) ) * (-0.5)
# PNSC = ( -160/9* S1 + 32/3.* S2 + 4/3 +
#           16*(11*NS+5*N-3)/(9* NS * N1S))*(-0.5)

# PPSA = ((5* NFI + 32* NFO + 49* NT+38* NS + 28* N + 8)
#           / (NM * NT * N1T * N2S) * 2   )
# PGGA = (- (2* NFI + 5* NFO + 8* NT + 7* NS- 2* N - 2)
#           * 8* S1 / (NMS * NS * N1S * N2S) -  67/9* S1 + 8/3
#           - 4* SSTR2P * (NS + N + 1) / (NM * N * N1 * N2)
#           + 2* S1 * SSTR2P - 4* SSCHLP + 0.5 * SSTR3P
#           + (457* NN + 2742* NE + 6040* NSE + 6098* NSI
#           + 1567* NFI - 2344* NFO - 1632* NT + 560* NS
#           + 1488* N + 576) / (18* NMS * NT * N1T * N2T))
# PGGB = ((38* NFO + 76* NT + 94* NS + 56* N + 12) *(-2)
#           / (9* NM * NS * N1S * N2)  +  20/9* S1  -  4/3)
# PGGC = ((2* NSI + 4* NFI + NFO - 10* NT - 5* NS - 4* N
#           - 4) * (-2) / (NM * NT * N1T * N2)  -  1)

# PPSTL = (-40/9 * 1/NM + 4/NT + 10/NS - 16/N
#           + 8/N1 + 112/9 * 1/N2 + 18/N1S
#           + 4/N1T + 16/3 * 1/N2S)

# PQQATL = (( -4 * S1 + 3 + 2/(N*N1) )
#           * ( 2*S2 - 2 * ZETA2 - (2*N + 1)/(NS*N1S) ))
# PQQBTL = (-80/9 * 1/NM + 8/NT + 12/NS - 12/N
#           + 8/N1T + 28/N1S - 4/N1 + 32/3 * 1/N2S
#           + 224/9 * 1/N2)

# PQGA = (S11 * (NS + N + 2)/(N * N1 * N2) + 1/NS - 5/3 * 1/N
#           - 1/(N * N1) - 2/N1S + 4/3 * 1/N1 + 4/N2S
#           - 4/3 * 1/N2)
# PQGB = (( - 2 * S11**2 + 2 * S11 + 10 * S21 )
#           * ( NS + N + 2 ) / ( N * N1 * N2 )
#           + 4 * S11
#           * ( -1/NS + 1/N + 1/(N*N1) + 2/N1S - 4/N2S )
#           - 2/NT + 5/NS - 12/N + 4/(NS*N1) - 12/(N*N1S)
#           - 6/(N*N1) + 4/N1T - 4/N1S + 23/N1 - 20/N2)
# PQGC = (( 2 * S11**2 - 10/3 * S11 - 6 * S21
#           + 1 * ( DPSI(N2/2,1) - DPSI(N1/2,1) ) - 6 * ZETA2 )
#           * ( NS + N + 2 ) / ( N * N1 * N2 )
#           - 4 * S11 *
#           ( -2/NS + 1/N + 1/(N*N1) + 4/N1S - 6/N2S )
#           - 40/9 * 1/NM + 4/NT + 8/3 * 1/NS
#           + 26/9 * 1/N - 8/(NS*N1S) + 22/3 * 1/(N*N1)
#           + 16/N1T + 68/3 * 1/N1S - 190/9 * 1/N1
#           + 8/(N1S*N2) - 4/N2S + 356/9 * 1/N2)

# PGQA = (( S1**2 - 3*S2 - 4 * ZETA2 )
#           * ( NS + N + 2 ) / ( NM * N * N1 )
#           + 2 * S1
#           * ( 4/NMS - 2/(NM*N) - 4/NS + 3/N1S - 1/N1 )
#           - 8/(NMS*N) + 8/(NM*NS) + 2/NT + 8/NS - 1/(2*N)
#           + 1/N1T - 5/2 * 1/N1S + 9/2 * 1/N1)

# PGQB = (( -1 * S1**2 + 5 * S2
#           - 0.5 * ( DPSI(N1/2,1) - DPSI(N/2,1) ) + ZETA2 )
#           * ( NS + N + 2 ) / ( NM * N * N1 )
#           + 2 * S1
#           * ( -2/NMS + 2/(NM*N) + 2/NS - 2/N1S + 1/N1 )
#           - 8/NMT + 6/NMS + 17/9 * 1/NM + 4/(NMS*N)
#           - 12/(NM*NS) - 8/NS + 5/N - 2/(NS*N1) - 2/N1T
#           - 7/N1S - 1/N1 - 8/3 * 1/N2S - 44/9 * 1/N2)

# PGGATL = (- 16/3 * 1/NMS + 80/9 * 1/NM + 8/NT
#           - 16/NS + 12/N + 8/N1T - 24/N1S + 4/N1
#           - 16/3 * 1/N2S - 224/9 * 1/N2)
# PGGBTL = S2 - 1/NMS + 1/NS - 1/N1S + 1/N2S - ZETA2

# PGGCTL = (- 8 * S1 * S2 + 8 * S1
#           * ( 1 / NMS - 1 / NS + 1 / N1S - 1 / N2S + ZETA2 )
#           + ( 8 * S2 - 8 * ZETA2 )
#           * ( 1 / NM - 1 / N + 1 / N1 - 1 / N2 + 11 / 12 )
#           - 8 / NMT + 22 / 3 * 1 / NMS - 8 / ( NMS * N )
#           - 8 / ( NM * NS ) - 8 / NT - 14 / 3 * 1 / NS
#           - 8 / N1T + 14 / 3 * 1 / N1S - 8 / ( N1S * N2 )
#           - 8 / ( N1 * N2S ) - 8 / N2T - 22 / 3 * 1 / N2S)

# PNSTL = (( - 4*S1 + 3 + 2/(N*N1) )
#           * ( 2*S2 - 2*ZETA2 - (2*N + 1)/(NS*N1S) ))


@nb.njit(cache=True)
def gamma_nsp(N, nf):
    NS = N * N
    NT = NS * N
    NFO = NT * N
    NFI = NFO * N
    NSI = NFI * N
    NSE = NSI * N
    NE = NSE * N
    NN = NE * N

    NM2 = N - 2
    NM = N - 1
    N1 = N + 1
    N2 = N + 2
    NMS = NM * NM
    NMT = NMS * NM
    N1S = N1 * N1
    N1T = N1S * N1
    N2S = N2 * N2
    N2T = N2S * N2

    S1 = w1.S1(N)
    S2 = w2.S2(N)

    N3 = N + 3
    N4 = N + 4
    N5 = N + 5
    N6 = N + 6

    S11 = S1 + 1 / N1
    S12 = S11 + 1 / N2
    S13 = S12 + 1 / N3
    S14 = S13 + 1 / N4
    S15 = S14 + 1 / N5
    S16 = S15 + 1 / N6
    S1M = S1 - 1 / N
    S21 = S2 + 1 / N1S
    S22 = S21 + 1 / N2S
    S2M = S2 - 1 / NS

    SPMOM = (
        1.0000 * (ZETA2 - S1 / N) / N
        - 0.9992 * (ZETA2 - S11 / N1) / N1
        + 0.9851 * (ZETA2 - S12 / N2) / N2
        - 0.9005 * (ZETA2 - S13 / N3) / N3
        + 0.6621 * (ZETA2 - S14 / N4) / N4
        - 0.3174 * (ZETA2 - S15 / N5) / N5
        + 0.0699 * (ZETA2 - S16 / N6) / N6
    )

    SLC = -5 / 8 * ZETA3
    SLV = -ZETA2 / 2 * (polygamma(N1 / 2, 0) - polygamma(N / 2, 0)) + S1 / NS + SPMOM
    SSCHLM = SLC - SLV
    SSTR2M = ZETA2 - polygamma(N1 / 2, 1)
    SSTR3M = 0.5 * polygamma(N1 / 2, 2) + ZETA3

    SSCHLP = SLC + SLV
    SSTR2P = ZETA2 - polygamma(N2 / 2, 1)
    SSTR3P = 0.5 * polygamma(N2 / 2, 2) + ZETA3

    DS2NM = -polygamma(NM2 / 2 + 1, 1) + polygamma(NM / 2 + 1, 1)
    DS2N = -polygamma(NM / 2 + 1, 1) + polygamma(N / 2 + 1, 1)
    DS2N1 = -polygamma(N / 2 + 1, 1) + polygamma(N1 / 2 + 1, 1)
    DS2N2 = -polygamma(N1 / 2 + 1, 1) + polygamma(N2 / 2 + 1, 1)

    PNPA = (
        16 * S1 * (2 * N + 1) / (NS * N1S)
        + 16 * (2 * S1 - 1 / (N * N1)) * (S2 - SSTR2P)
        + 64 * SSCHLP
        + 24 * S2
        - 3
        - 8 * SSTR3P
        - 8 * (3 * NT + NS - 1) / (NT * N1T)
        - 16 * (2 * NS + 2 * N + 1) / (NT * N1T)
    ) * (-0.5)
    PNSB = (
        S1 * (536 / 9 + 8 * (2 * N + 1) / (NS * N1S))
        - (16 * S1 + 52 / 3 - 8 / (N * N1)) * S2
        - 43 / 6
        - (151 * NFO + 263 * NT + 97 * NS + 3 * N + 9) * 4 / (9 * NT * N1T)
    ) * (-0.5)
    PNSC = (
        -160 / 9 * S1
        + 32 / 3.0 * S2
        + 4 / 3
        + 16 * (11 * NS + 5 * N - 3) / (9 * NS * N1S)
    ) * (-0.5)
    PNSTL = (-4 * S1 + 3 + 2 / (N * N1)) * (
        2 * S2 - 2 * ZETA2 - (2 * N + 1) / (NS * N1S)
    )

    result = (
        constants.CF
        * (
            (constants.CF - constants.CA / 2) * PNPA
            + constants.CA * PNSB
            + (1 / 2) * nf * PNSC
        )
        + constants.CF**2 * PNSTL * 4
    )
    return -result


@nb.njit(cache=True)
def gamma_nsm(N, nf):
    NS = N * N
    NT = NS * N
    NFO = NT * N
    NFI = NFO * N
    NSI = NFI * N
    NSE = NSI * N
    NE = NSE * N
    NN = NE * N

    NM2 = N - 2
    NM = N - 1
    N1 = N + 1
    N2 = N + 2
    NMS = NM * NM
    NMT = NMS * NM
    N1S = N1 * N1
    N1T = N1S * N1
    N2S = N2 * N2
    N2T = N2S * N2

    S1 = w1.S1(N)
    S2 = w2.S2(N)

    N3 = N + 3
    N4 = N + 4
    N5 = N + 5
    N6 = N + 6

    S11 = S1 + 1 / N1
    S12 = S11 + 1 / N2
    S13 = S12 + 1 / N3
    S14 = S13 + 1 / N4
    S15 = S14 + 1 / N5
    S16 = S15 + 1 / N6
    S1M = S1 - 1 / N
    S21 = S2 + 1 / N1S
    S22 = S21 + 1 / N2S
    S2M = S2 - 1 / NS

    SPMOM = (
        1.0000 * (ZETA2 - S1 / N) / N
        - 0.9992 * (ZETA2 - S11 / N1) / N1
        + 0.9851 * (ZETA2 - S12 / N2) / N2
        - 0.9005 * (ZETA2 - S13 / N3) / N3
        + 0.6621 * (ZETA2 - S14 / N4) / N4
        - 0.3174 * (ZETA2 - S15 / N5) / N5
        + 0.0699 * (ZETA2 - S16 / N6) / N6
    )

    SLC = -5 / 8 * ZETA3
    SLV = -ZETA2 / 2 * (polygamma(N1 / 2, 0) - polygamma(N / 2, 0)) + S1 / NS + SPMOM
    SSCHLM = SLC - SLV
    SSTR2M = ZETA2 - polygamma(N1 / 2, 1)
    SSTR3M = 0.5 * polygamma(N1 / 2, 2) + ZETA3

    SSCHLP = SLC + SLV
    SSTR2P = ZETA2 - polygamma(N2 / 2, 1)
    SSTR3P = 0.5 * polygamma(N2 / 2, 2) + ZETA3

    DS2NM = -polygamma(NM2 / 2 + 1, 1) + polygamma(NM / 2 + 1, 1)
    DS2N = -polygamma(NM / 2 + 1, 1) + polygamma(N / 2 + 1, 1)
    DS2N1 = -polygamma(N / 2 + 1, 1) + polygamma(N1 / 2 + 1, 1)
    DS2N2 = -polygamma(N1 / 2 + 1, 1) + polygamma(N2 / 2 + 1, 1)

    PNMA = (
        16 * S1 * (2 * N + 1) / (NS * N1S)
        + 16 * (2 * S1 - 1 / (N * N1)) * (S2 - SSTR2M)
        + 64 * SSCHLM
        + 24 * S2
        - 3
        - 8 * SSTR3M
        - 8 * (3 * NT + NS - 1) / (NT * N1T)
        + 16 * (2 * NS + 2 * N + 1) / (NT * N1T)
    ) * (-0.5)
    PNSB = (
        S1 * (536 / 9 + 8 * (2 * N + 1) / (NS * N1S))
        - (16 * S1 + 52 / 3 - 8 / (N * N1)) * S2
        - 43 / 6
        - (151 * NFO + 263 * NT + 97 * NS + 3 * N + 9) * 4 / (9 * NT * N1T)
    ) * (-0.5)
    PNSC = (
        -160 / 9 * S1
        + 32 / 3.0 * S2
        + 4 / 3
        + 16 * (11 * NS + 5 * N - 3) / (9 * NS * N1S)
    ) * (-0.5)
    PNSTL = (-4 * S1 + 3 + 2 / (N * N1)) * (
        2 * S2 - 2 * ZETA2 - (2 * N + 1) / (NS * N1S)
    )

    result = (
        constants.CF
        * (
            (constants.CF - constants.CA / 2) * PNMA
            + constants.CA * PNSB
            + (1 / 2) * nf * PNSC
        )
        + constants.CF**2 * PNSTL * 4
    )
    return -result


@nb.njit(cache=True)
def gamma_singlet(N, nf):
    NS = N * N
    NT = NS * N
    NFO = NT * N
    NFI = NFO * N
    NSI = NFI * N
    NSE = NSI * N
    NE = NSE * N
    NN = NE * N

    NM2 = N - 2
    NM = N - 1
    N1 = N + 1
    N2 = N + 2
    NMS = NM * NM
    NMT = NMS * NM
    N1S = N1 * N1
    N1T = N1S * N1
    N2S = N2 * N2
    N2T = N2S * N2

    S1 = w1.S1(N)
    S2 = w2.S2(N)

    N3 = N + 3
    N4 = N + 4
    N5 = N + 5
    N6 = N + 6

    S11 = S1 + 1 / N1
    S12 = S11 + 1 / N2
    S13 = S12 + 1 / N3
    S14 = S13 + 1 / N4
    S15 = S14 + 1 / N5
    S16 = S15 + 1 / N6
    S1M = S1 - 1 / N
    S21 = S2 + 1 / N1S
    S22 = S21 + 1 / N2S
    S2M = S2 - 1 / NS

    SPMOM = (
        1.0000 * (ZETA2 - S1 / N) / N
        - 0.9992 * (ZETA2 - S11 / N1) / N1
        + 0.9851 * (ZETA2 - S12 / N2) / N2
        - 0.9005 * (ZETA2 - S13 / N3) / N3
        + 0.6621 * (ZETA2 - S14 / N4) / N4
        - 0.3174 * (ZETA2 - S15 / N5) / N5
        + 0.0699 * (ZETA2 - S16 / N6) / N6
    )

    SLC = -5 / 8 * ZETA3
    SLV = -ZETA2 / 2 * (polygamma(N1 / 2, 0) - polygamma(N / 2, 0)) + S1 / NS + SPMOM
    SSCHLM = SLC - SLV
    SSTR2M = ZETA2 - polygamma(N1 / 2, 1)
    SSTR3M = 0.5 * polygamma(N1 / 2, 2) + ZETA3

    SSCHLP = SLC + SLV
    SSTR2P = ZETA2 - polygamma(N2 / 2, 1)
    SSTR3P = 0.5 * polygamma(N2 / 2, 2) + ZETA3

    DS2NM = -polygamma(NM2 / 2 + 1, 1) + polygamma(NM / 2 + 1, 1)
    DS2N = -polygamma(NM / 2 + 1, 1) + polygamma(N / 2 + 1, 1)
    DS2N1 = -polygamma(N / 2 + 1, 1) + polygamma(N1 / 2 + 1, 1)
    DS2N2 = -polygamma(N1 / 2 + 1, 1) + polygamma(N2 / 2 + 1, 1)

    PNMA = (
        16 * S1 * (2 * N + 1) / (NS * N1S)
        + 16 * (2 * S1 - 1 / (N * N1)) * (S2 - SSTR2M)
        + 64 * SSCHLM
        + 24 * S2
        - 3
        - 8 * SSTR3M
        - 8 * (3 * NT + NS - 1) / (NT * N1T)
        + 16 * (2 * NS + 2 * N + 1) / (NT * N1T)
    ) * (-0.5)
    PNPA = (
        16 * S1 * (2 * N + 1) / (NS * N1S)
        + 16 * (2 * S1 - 1 / (N * N1)) * (S2 - SSTR2P)
        + 64 * SSCHLP
        + 24 * S2
        - 3
        - 8 * SSTR3P
        - 8 * (3 * NT + NS - 1) / (NT * N1T)
        - 16 * (2 * NS + 2 * N + 1) / (NT * N1T)
    ) * (-0.5)

    PNSB = (
        S1 * (536 / 9 + 8 * (2 * N + 1) / (NS * N1S))
        - (16 * S1 + 52 / 3 - 8 / (N * N1)) * S2
        - 43 / 6
        - (151 * NFO + 263 * NT + 97 * NS + 3 * N + 9) * 4 / (9 * NT * N1T)
    ) * (-0.5)
    PNSC = (
        -160 / 9 * S1
        + 32 / 3.0 * S2
        + 4 / 3
        + 16 * (11 * NS + 5 * N - 3) / (9 * NS * N1S)
    ) * (-0.5)

    PPSA = (
        (5 * NFI + 32 * NFO + 49 * NT + 38 * NS + 28 * N + 8)
        / (NM * NT * N1T * N2S)
        * 2
    )
    PGGA = (
        -(2 * NFI + 5 * NFO + 8 * NT + 7 * NS - 2 * N - 2)
        * 8
        * S1
        / (NMS * NS * N1S * N2S)
        - 67 / 9 * S1
        + 8 / 3
        - 4 * SSTR2P * (NS + N + 1) / (NM * N * N1 * N2)
        + 2 * S1 * SSTR2P
        - 4 * SSCHLP
        + 0.5 * SSTR3P
        + (
            457 * NN
            + 2742 * NE
            + 6040 * NSE
            + 6098 * NSI
            + 1567 * NFI
            - 2344 * NFO
            - 1632 * NT
            + 560 * NS
            + 1488 * N
            + 576
        )
        / (18 * NMS * NT * N1T * N2T)
    )
    PGGB = (
        (38 * NFO + 76 * NT + 94 * NS + 56 * N + 12) * (-2) / (9 * NM * NS * N1S * N2)
        + 20 / 9 * S1
        - 4 / 3
    )
    PGGC = (2 * NSI + 4 * NFI + NFO - 10 * NT - 5 * NS - 4 * N - 4) * (-2) / (
        NM * NT * N1T * N2
    ) - 1

    PPSTL = (
        -40 / 9 * 1 / NM
        + 4 / NT
        + 10 / NS
        - 16 / N
        + 8 / N1
        + 112 / 9 * 1 / N2
        + 18 / N1S
        + 4 / N1T
        + 16 / 3 * 1 / N2S
    )

    PQQATL = (-4 * S1 + 3 + 2 / (N * N1)) * (
        2 * S2 - 2 * ZETA2 - (2 * N + 1) / (NS * N1S)
    )
    PQQBTL = (
        -80 / 9 * 1 / NM
        + 8 / NT
        + 12 / NS
        - 12 / N
        + 8 / N1T
        + 28 / N1S
        - 4 / N1
        + 32 / 3 * 1 / N2S
        + 224 / 9 * 1 / N2
    )

    PQGA = (
        S11 * (NS + N + 2) / (N * N1 * N2)
        + 1 / NS
        - 5 / 3 * 1 / N
        - 1 / (N * N1)
        - 2 / N1S
        + 4 / 3 * 1 / N1
        + 4 / N2S
        - 4 / 3 * 1 / N2
    )
    PQGB = (
        (-2 * S11**2 + 2 * S11 + 10 * S21) * (NS + N + 2) / (N * N1 * N2)
        + 4 * S11 * (-1 / NS + 1 / N + 1 / (N * N1) + 2 / N1S - 4 / N2S)
        - 2 / NT
        + 5 / NS
        - 12 / N
        + 4 / (NS * N1)
        - 12 / (N * N1S)
        - 6 / (N * N1)
        + 4 / N1T
        - 4 / N1S
        + 23 / N1
        - 20 / N2
    )
    PQGC = (
        (
            2 * S11**2
            - 10 / 3 * S11
            - 6 * S21
            + 1 * (polygamma(N2 / 2, 1) - polygamma(N1 / 2, 1))
            - 6 * ZETA2
        )
        * (NS + N + 2)
        / (N * N1 * N2)
        - 4 * S11 * (-2 / NS + 1 / N + 1 / (N * N1) + 4 / N1S - 6 / N2S)
        - 40 / 9 * 1 / NM
        + 4 / NT
        + 8 / 3 * 1 / NS
        + 26 / 9 * 1 / N
        - 8 / (NS * N1S)
        + 22 / 3 * 1 / (N * N1)
        + 16 / N1T
        + 68 / 3 * 1 / N1S
        - 190 / 9 * 1 / N1
        + 8 / (N1S * N2)
        - 4 / N2S
        + 356 / 9 * 1 / N2
    )

    PGQA = (
        (S1**2 - 3 * S2 - 4 * ZETA2) * (NS + N + 2) / (NM * N * N1)
        + 2 * S1 * (4 / NMS - 2 / (NM * N) - 4 / NS + 3 / N1S - 1 / N1)
        - 8 / (NMS * N)
        + 8 / (NM * NS)
        + 2 / NT
        + 8 / NS
        - 1 / (2 * N)
        + 1 / N1T
        - 5 / 2 * 1 / N1S
        + 9 / 2 * 1 / N1
    )

    PGQB = (
        (
            -1 * S1**2
            + 5 * S2
            - 0.5 * (polygamma(N1 / 2, 1) - polygamma(N / 2, 1))
            + ZETA2
        )
        * (NS + N + 2)
        / (NM * N * N1)
        + 2 * S1 * (-2 / NMS + 2 / (NM * N) + 2 / NS - 2 / N1S + 1 / N1)
        - 8 / NMT
        + 6 / NMS
        + 17 / 9 * 1 / NM
        + 4 / (NMS * N)
        - 12 / (NM * NS)
        - 8 / NS
        + 5 / N
        - 2 / (NS * N1)
        - 2 / N1T
        - 7 / N1S
        - 1 / N1
        - 8 / 3 * 1 / N2S
        - 44 / 9 * 1 / N2
    )

    PGGATL = (
        -16 / 3 * 1 / NMS
        + 80 / 9 * 1 / NM
        + 8 / NT
        - 16 / NS
        + 12 / N
        + 8 / N1T
        - 24 / N1S
        + 4 / N1
        - 16 / 3 * 1 / N2S
        - 224 / 9 * 1 / N2
    )
    PGGBTL = S2 - 1 / NMS + 1 / NS - 1 / N1S + 1 / N2S - ZETA2

    PGGCTL = (
        -8 * S1 * S2
        + 8 * S1 * (1 / NMS - 1 / NS + 1 / N1S - 1 / N2S + ZETA2)
        + (8 * S2 - 8 * ZETA2) * (1 / NM - 1 / N + 1 / N1 - 1 / N2 + 11 / 12)
        - 8 / NMT
        + 22 / 3 * 1 / NMS
        - 8 / (NMS * N)
        - 8 / (NM * NS)
        - 8 / NT
        - 14 / 3 * 1 / NS
        - 8 / N1T
        + 14 / 3 * 1 / N1S
        - 8 / (N1S * N2)
        - 8 / (N1 * N2S)
        - 8 / N2T
        - 22 / 3 * 1 / N2S
    )

    PNSTL = (-4 * S1 + 3 + 2 / (N * N1)) * (
        2 * S2 - 2 * ZETA2 - (2 * N + 1) / (NS * N1S)
    )

    el1 = -(
        (
            constants.CF
            * (
                (constants.CF - constants.CA / 2) * PNPA
                + constants.CA * PNSB
                + (1 / 2) * nf * PNSC
            )
            + constants.CF**2 * PNSTL * 4
        )
        + (1 / 2) * nf * constants.CF * PPSTL * 4
    )
    el2 = -(constants.CF**2 * PGQA + constants.CF * constants.CA * PGQB) * 4 * 2 * nf
    el3 = -(
        (
            8 / 3 * ((1 / 2) * nf) ** 2 * PQGA
            + constants.CF * (1 / 2) * nf * PQGB
            + constants.CA * (1 / 2) * nf * PQGC
        )
        * 4
        / (2 * nf)
    )
    el4 = -(
        (
            constants.CA * constants.CA * PGGA
            + (1 / 2) * nf * (constants.CA * PGGB + constants.CF * PGGC)
        )
        * 4
        + constants.CF * (1 / 2) * nf * 4 * PGGATL
        + constants.CA * (1 / 2) * nf * 4 * (-8 / 3) * PGGBTL
        + constants.CA * constants.CA * 4 * PGGCTL
    )
    result = np.array(
        [
            [el1, el2],
            [el3, el4],
        ],
        np.complex_,
    )
    return result
