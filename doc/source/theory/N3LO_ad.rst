N3LO Anomalous Dimensions
=========================

The |N3LO| |QCD| anomalous dimensions :math:`\gamma^{(3)}` are not yet fully known,
since they rely on the calculation of 4-loop DIS integrals.
Moreover the analytical structure of these function is already known to be complicated
since in Mellin space it will included harmonics sum up to weight 7, for which an
analytical contribution is not available.

Here we describe the various assumptions and limits used in order to reconstruct a parametrization
that can approximate their contribution.
In particular we will take advantage of some known physical constrain,
such as large-x, small-x limits  and sum rules in order to make our reconstruction reasonable.

Generally we remark that the large-x limit correspond to large-N in Mellin space
where the leading contribution comes from the harmonics :math:`S_1(N)`,
while  the small-x region corresponds to poles at :math:`N=0,1` depending on the type of
divergence.

In any case |N3LO| |DGLAP| evolution at small-x, especially for singlet-like PDFs, will not be reliable
until the splitting function resummation will not be available up to NNLL.

Non Singlet sector
------------------

In the non singlet sector we construct a parametrization for
:math:`\gamma_{ns,-}^{(3)},\gamma_{ns,-}^{(3)},\gamma_{ns,s}^{(3)}` where:

    .. math ::
        \gamma_{ns,s}^{(3)} =  \gamma_{ns,v}^{(3)} - \gamma_{ns,-}^{(3)}

In particular making explicitly the dependence on :math:`n_f`the non-singlet anomalous dimensions include:

    .. list-table:: Non singlet 4-loop Anomalous Dimensions
        :header-rows: 1

        *   -
            - :math:`n_{f}^0`
            - :math:`n_{f}^1`
            - :math:`n_{f}^2`
            - :math:`n_{f}^3`

        *   - :math:`\gamma_{ns,-}^{(3)}`
            - |T|
            - |T|
            - |T|
            - |T|

        *   - :math:`\gamma_{ns,+}^{(3)}`
            - |T|
            - |T|
            - |T|
            - |T|

        *   - :math:`\gamma_{ns,s}^{(3)}`
            -
            - |T|
            - |T|
            -

Where:

    * the part proportional to :math:`nf^3` is common for :math:`\gamma_{ns,+}^{(3)},\gamma_{ns,-}^{(3)},\gamma_{ns,v}^{(3)}`
      and is exact, see :cite:`Davies:2016jie` (Eq. 3.6).

    * In :math:`\gamma_{ns,s}^{(3)}` the part proportional to :math:`nf^2`
      is exact see :cite:`Davies:2016jie` (Eq. 3.5).

    * In :math:`\gamma_{ns,s}^{(3)}` the part proportional to :math:`nf^1` is
      parametrized in x-space and copied from :cite:`Moch:2017uml`, (Eq. 4.19, 4.20).

    *   The remaining contributions include the following constrain:

        -   The small-x limit, given in the large :math:`N_c` approximation by
            :cite:`Davies:2022ofz` (see Eq. 3.3, 3.8, 3.9, 3.10).
            Note the expressions are evaluated with the exact values of the |QCD|
            to better agree with the :cite:`Moch:2017uml` parametrization.
            This parts contains functions :math:`\ln(x)^k` with :math:`k=1,..,6`
            Which correspond to :math:`1/N^7,...1/N^2` in Mellin space.

        -   The large-N limit see :cite:`Moch:2017uml` (Eq. 2.17):

            .. math ::
                \gamma_{ns} \approx A_4 S_1(N) - B_4 + C_4 \frac{S_1(N)}{N} - (D_4 + \frac{1}{2} A_4) \frac{1}{N} + \mathcal{O}(\frac{\ln(N)^k}{N^2})

            This limit is common for all :math:`\\gamma_{ns,+}^{(3)},\\gamma_{ns,-}^{(3)},\\gamma_{ns,v}^{(3)}`.
            The coefficient :math:`A_4` can be obtained from the |QCD| cusp calculation
            :cite:`Henn:2019swt` while the :math:`B_4` is fixed by the integral of the 4-loop splitting function
            and :math:`C_4,D_4` depends on lower order splitting functions.

        -   The 8 lowest N moments provided in :cite:`Moch:2017uml`. In particular
            we have that :math:`\gamma_{ns,s}(1)=\gamma_{ns,-}(1)=0` which correspond
            to quark number conservation.

        -   The difference between the known moments and the known limits is parametrized
            in Mellin space. The basis includes:

            .. list-table::
                :header-rows: 1

                *   - N-space
                    - x-space
                *   - 1
                    - :math:`\delta(1-x)`
                *   - :math:`\mathcal{M}[(1-x)\ln(1-x)]`
                    - :math:`(1-x)\ln(1-x)`
                *   - :math:`\mathcal{M}[(1-x)\ln^2(1-x)]`
                    - :math:`(1-x)\ln^2(1-x)`
                *   - :math:`\mathcal{M}[(1-x)\ln^3(1-x)]`
                    - :math:`(1-x)\ln^3(1-x)`
                *   - :math:`\frac{S_1(N)}{N^2}`
                    - :math:`- Li_2(x) + \zeta_2`

            which model the sub-leading differences in the large-N limit, and:

            .. list-table::
                :header-rows: 1

                *   - N-space
                    - x-space
                *   - :math:`\frac{1}{(N+1)^2}`
                    - :math:`x\ln(x)`
                *   - :math:`\frac{1}{(N+1)^3}`
                    - :math:`\frac{x}{2}\ln^2(x)`

            to help the convergence of the small-N moments. Finally we add a polynomial part
            :math:`x^2(3)` which corresponds to simple poles at :math:`N=-2,-3`
            respectively for :math:`\gamma_{ns,+},\gamma_{ns,-}`.

            Note that the constant coefficient is included in the fit, following the procedure done
            in :cite:`Moch:2017uml` (section 4), to achieve a better accuracy.
            It is check that this contribution is much more smaller than the values of :math:`B_4`.
