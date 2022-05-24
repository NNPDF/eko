# -*- coding: utf-8 -*-
"""
This file contains the |N3LO| Altarelli-Parisi splitting kernels.


For the non singlet anomalous dimensions:
    *   The part proportional to :math:`nf^3`,
        is common for :math:`\\gamma_{ns,+}^{(3)},\\gamma_{ns,-}^{(3)},\\gamma_{ns,v}^{(3)}`
        is copied exact from :cite:`Davies:2016jie` (Eq. 3.6).

    *   In :math:`\\gamma_{ns,s}^{(3)}`: the part proportional to :math:`nf^2` is exact.
    *   In :math:`\\gamma_{ns,s}^{(3)}`: the part proportional to :math:`nf^1` is
        copied from :cite:`Moch:2017uml`, `gNSv.gamma_nss_nf1`.

    *   The difference between :math:`\\gamma_{ns,+}^{(3)}-\\gamma_{ns,-}^{(3)}` proportional to
        :math:`nf^2` is exact, see `gNSp.delta_B3`.

    *   The remaining contributions are all fitted and includes:
        -   The small-x limit, given in the large :math:`n_c` approximation by
            :cite:`Davies:2022ofz` (see Eq. 3.3, 3.8, 3.9, 3.10).
        -   The large-N limit see :cite:`Moch:2017uml` (Eq. 2.17), where :math:`\\ln(N)+\\gamma_{E}`
            is replaced by :math:`S1`.
            This limit is common for all
            :math:`\\gamma_{ns,+}^{(3)},\\gamma_{ns,-}^{(3)},\\gamma_{ns,v}^{(3)}`.
        -   The 8 lowest N moments provided in :cite:`Moch:2017uml`.
        -   The difference between the given moments and the known limits is fitted
            using a basis containing:
                :math:`1,1/n,1/(n+1),1/(n+2),1/(n+3),1/(n+4),S1/(n+1),S1/n^2`

The large-N expression are based on the 4-loop QCD-cusp calculation :cite:`Henn:2019swt`.
"""
from .gNSm import gamma_nsm
from .gNSp import gamma_nsp
from .gNSv import gamma_nsv
