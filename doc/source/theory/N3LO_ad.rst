N3LO Anomalous Dimensions
=========================

The |N3LO| |QCD| anomalous dimensions :math:`\gamma^{(3)}` are not yet fully known,
since they rely on the calculation of 4-loop |DIS| integrals.
Moreover, the analytical structure of these function is already known to be complicated
since in Mellin space they include harmonics sum up to weight 7, for which an
analytical expression is not available.

Here, we describe the various assumptions and limits used in order to reconstruct a parametrization
that can approximate their contribution.
In particular we take advantage of some known physical constrains,
such as the large-x limit, the small-x limit, and sum rules, in order to make our reconstruction reasonable.

Generally, we remark that the large-x limit correspond to large-N in Mellin space
where the leading contribution comes from the harmonics :math:`S_1(N)`,
while the small-x region corresponds to poles at :math:`N=0,1` depending on the type of
divergence.

In any case |N3LO| |DGLAP| evolution at small-x, especially for singlet-like PDFs, is not reliable
until the splitting function resummation is available up to |NNLL|.

Non-singlet sector
------------------

In the non-singlet sector we construct a parametrization for
:math:`\gamma_{ns,-}^{(3)},\gamma_{ns,-}^{(3)},\gamma_{ns,s}^{(3)}` where:

    .. math ::
        \gamma_{ns,s}^{(3)} = \gamma_{ns,v}^{(3)} - \gamma_{ns,-}^{(3)}

In particular, making explicitly the dependence on :math:`n_f`, the non-singlet anomalous dimensions include
the following terms:

    .. list-table:: non-singlet 4-loop Anomalous Dimensions
        :align: center
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

Some of these parts are known analytically exactly (:math:`\propto n_f^2,n_f^3`),
while others are available only in the large :math:`N_c` limit (:math:`\propto n_f^0,n_f^1`).
In |EKO| they are implemented as follows:

    * the part proportional to :math:`n_f^3` is common for :math:`\gamma_{ns,+}^{(3)},\gamma_{ns,-}^{(3)},\gamma_{ns,v}^{(3)}`
      and is exact :cite:`Davies:2016jie` (Eq. 3.6).

    * In :math:`\gamma_{ns,s}^{(3)}` the part proportional to :math:`n_f^2`
      is exact :cite:`Davies:2016jie` (Eq. 3.5).

    * In :math:`\gamma_{ns,s}^{(3)}` the part proportional to :math:`n_f^1` is
      parametrized in x-space and copied from :cite:`Moch:2017uml` (Eq. 4.19, 4.20).

    * The remaining contributions include the following constrains:

        -   The small-x limit, given in the large :math:`N_c` approximation by
            :cite:`Davies:2022ofz` (see Eq. 3.3, 3.8, 3.9, 3.10) and coming
            from small-x resummation.
            This part contains the so called double logarithms:

            .. math ::
                \gamma_{ns} \approx \sum_{k=1}^{6} c^{(k)} \ln^k(x) \quad \text{with:}  \quad \mathcal{M}[\ln^k(x)] = \frac{1}{N^{k+1}}


            Note the expressions are evaluated with the exact values of the |QCD|
            Casimir invariants, to better agree with the :cite:`Moch:2017uml` parametrization.

        -   The large-N limit :cite:`Moch:2017uml`, which reads (Eq. 2.17):

            .. math ::
                \gamma_{ns} \approx A^{(f)}_4 S_1(N) - B_4 + C_4 \frac{S_1(N)}{N} - D_4 \frac{1}{N}

            This limit is common for all :math:`\gamma_{ns,+}^{(3)},\gamma_{ns,-}^{(3)},\gamma_{ns,v}^{(3)}`.
            The coefficient :math:`A^{(f)}_4`, being related to the twist-2 spin-N operators,
            can be obtained from the |QCD| cusp calculation
            :cite:`Henn:2019swt`, while the :math:`B_4` is fixed by the integral of the 4-loop splitting function
            and has been firstly computed in :cite:`Moch:2017uml` in the large :math:`n_c` limit.
            More recently :cite:`Duhr:2022cob`, it has been determined  in the full color expansion
            by computing various |N3LO| cross sections in the soft limit.
            :math:`C_4,D_4` instead can be computed directly from lower order splitting functions.
            From large-x resummation :cite:`Davies:2016jie`, it is possible to infer further constrains
            on sub-leading terms :math:`\frac{\ln^k(N)}{N^2}`, since the non-singlet splitting
            functions contain only terms :math:`(1-x)^a\ln^k(1-x)` with :math:`a \ge 1`.

        -   The 8 lowest odd or even N moments provided in :cite:`Moch:2017uml`, where
            from quark number conservation we can trivially obtain:
            :math:`\gamma_{ns,s}(1)=\gamma_{ns,-}(1)=0`.

        -   The difference between the known moments and the known limits is parametrized
            in Mellin space. The basis includes:

            .. list-table:: :math:`\gamma_{ns,\pm}^{(3)}` parametrization basis
                :align: center
                :header-rows: 1

                *   - x-space
                    - N-space
                *   - :math:`\delta(1-x)`
                    - 1
                *   - :math:`(1-x)\ln(1-x)`
                    - :math:`\mathcal{M}[(1-x)\ln(1-x)]`
                *   - :math:`(1-x)\ln^2(1-x)`
                    - :math:`\mathcal{M}[(1-x)\ln^2(1-x)]`
                *   - :math:`(1-x)\ln^3(1-x)`
                    - :math:`\mathcal{M}[(1-x)\ln^3(1-x)]`
                *   - :math:`- \rm{Li_2}(x) + \zeta_2`
                    - :math:`\frac{S_1(N)}{N^2}`
                *   - :math:`x\ln(x)`
                    - :math:`\frac{1}{(N+1)^2}`
                *   - :math:`\frac{x}{2}\ln^2(x)`
                    - :math:`\frac{1}{(N+1)^3}`
                *   - :math:`x^{2}, x^{3}`
                    - :math:`\frac{1}{(N+2)},\frac{1}{(N+3)}`

            The first five functions model the sub-leading differences in the :math:`N\to \infty` limit,
            while the last three help the convergence in the small-N region. Finally, we add a polynomial part
            :math:`x^{2}` or :math:`x^{3}` respectively for :math:`\gamma_{ns,+},\gamma_{ns,-}`.
            For large-N we have the limit:

                .. math ::
                    \mathcal{M}[(1-x)\ln^k(1-x)] \approx \frac{S_1^k(N)}{N^2}

            Note that the constant coefficient is included in the fit, following the procedure done
            in :cite:`Moch:2017uml` (section 4), to achieve a better accuracy.
            It is checked that this contribution is much more smaller than the values of :math:`B_4`.

Singlet sector
--------------

In the singlet sector we construct a parametrization for
:math:`\gamma_{gg}^{(3)},\gamma_{gq}^{(3)},\gamma_{qg}^{(3)},\gamma_{qq}^{(3)}` where:

    .. math ::
        \gamma_{qq}^{(3)} = \gamma_{ns,+}^{(3)} + \gamma_{qq,ps}^{(3)}

In particular, making explicitly the dependence on :math:`n_f`, the singlet anomalous dimensions include
the following terms:

    .. list-table:: singlet 4-loop Anomalous Dimensions
        :align: center
        :header-rows: 1

        *   -
            - :math:`n_{f}^0`
            - :math:`n_{f}^1`
            - :math:`n_{f}^2`
            - :math:`n_{f}^3`


        *   - :math:`\gamma_{gg}^{(3)}`
            - |T|
            - |T|
            - |T|
            - |T|

        *   - :math:`\gamma_{gq}^{(3)}`
            - |T|
            - |T|
            - |T|
            - |T|

        *   - :math:`\gamma_{qg}^{(3)}`
            -
            - |T|
            - |T|
            - |T|

        *   - :math:`\gamma_{qq,ps}^{(3)}`
            -
            - |T|
            - |T|
            - |T|

The parts proportional to :math:`n_f^3` are known analytically
:cite:`Davies:2016jie` and have been included so far.
For :math:`\gamma_{qq,ps}` and :math:`\gamma_{gq}` also the component
proportional to :math:`n_f^2` has been computed in :cite:`Gehrmann:2023cqm`
and :cite:`Falcioni:2023tzp` respectively and it's used in our code
through an approximations obtained with 30 moments.

The other parts are approximated using some known limits:

    *   The small-x limit, given in the large :math:`N_c` approximation by
        :cite:`Davies:2022ofz` (see Eq. 5.9, 5.10, 5.11, 5.12) and coming
        from small-x resummation of double-logarithms which fix the leading terms
        for the pole at :math:`N=0`:

            .. math ::
                \gamma_{ij} \approx c^{(6)}_{ij} \ln^6(x) + c^{(5)}_{ij} \ln^5(x) + c^{(4)}_{ij} \ln^5(x) + \dots \quad \text{with:}  \quad  \mathcal{M}[\ln^k(x)] = \frac{1}{N^{k+1}}

    *   The small-x limit, coming from |BFKL| resummation
        :cite:`Bonvini:2018xvt` (see Eq. 2.32, 2.20b, 2.21a, 2.21b)
        which fix the leading terms (|LL|, |NLL|) for the pole at :math:`N=1`:

            .. math ::
                \gamma_{ij} \approx d^{(3)}_{ij} \frac{\ln^3(x)}{x} + d^{(2)}_{ij} \frac{\ln^2(x)}{x} + \dots \quad \text{with:}  \quad  \mathcal{M}[\frac{\ln^k(x)}{x}] = \frac{1}{(N-1)^{k+1}}

        Note that in principle also the term :math:`\frac{\ln^6(x)}{x}` could be present at |N3LO|,
        but they are vanishing.
        These terms are way larger than the previous ones in the small-x limit and
        are effectively determining the raise of the splitting functions at small-x.
        In particular only the expansion for :math:`\gamma_{gg}^{(3)}` is known at |NLL|.
        |LL| terms respect the representation symmetry :

            .. math ::
                \gamma_{gq} & \approx \frac{C_F}{C_A} \gamma_{gg}  \\
                \gamma_{qq,ps} & \approx \frac{C_F}{C_A} \gamma_{qg} \\


    *   The large-x limit of the singlet splitting function is different for the diagonal part
        and the off-diagonal.
        It is known that :cite:`Albino:2000cp,Moch:2021qrk` the diagonal terms diverge in N-space as:

            .. math ::
                \gamma_{kk} \approx A^{(r)}_4 S_1(N) + B^{(r)}_4 + C^{(r)}_4 \frac{S_1(N)}{N} + \mathcal{O}(\frac{1}{N})

        Where again the coefficient :math:`A^{(r)}_4` is the |QCD| cusp anomalous dimension for the adjoint or fundamental representation,
        the coefficient :math:`B^{(r)}_4` has been extracted from soft anomalous dimensions :cite:`Duhr:2022cob`.
        and :math:`C^{(r)}_4`can be estimate from lower orders :cite:`Dokshitzer:2005bf`.
        However, :math:`\gamma_{qq,ps}^{(3)}` do not constrain any divergence at large-x or constant term so its expansion starts as
        :math:`\mathcal{O}(\frac{1}{N^2})`.
        The off-diagonal do not contain any +-distributions or delta distributions but can include divergent logarithms
        of the type :cite:`Soar:2009yh`:

            .. math ::
                \ln^k(1-x) \quad k=1,..,6

        where also in this case the term :math:`k=6` vanish. The values of the coefficient for :math:`k=4,5`
        can be guessed from the lower order splitting functions. These logarithms are not present in the diagonal
        splitting function, which can include at most terms :math:`(1-x)\ln^4(1-x)`. While for :math:`\gamma_{gg}`
        these contributions are beyond the accuracy of our implementation, they are relevant for :math:`\gamma_{qq,ps}`.
        At large-x we have :cite:`Soar:2009yh`:

            .. math ::
                \gamma_{qq,ps} \approx (1-x)[c_{4} \ln^4(1-x) + c_{3} \ln^3(1-x)] + \mathcal{O}((1-x)\ln^2(1-x))


    *   The 4 lowest even N moments provided in :cite:`Moch:2021qrk`, where we can use momentum conservation
        to fix:

            .. math ::
                & \gamma_{qg}(2) + \gamma_{gg}(2) = 0 \\
                & \gamma_{qq}(2) + \gamma_{gq}(2) = 0 \\

        For :math:`\gamma_{qq,ps}, \gamma_{qg}` other 6 additional moments are available :cite:`Falcioni:2023luc,Falcioni:2023vqq`.
        making the parametrization of this splitting function much more accurate.

The difference between the known moments and the known limits is parametrized
in Mellin space using different basis, in order to estimate the uncertainties of
our determination.


Uncertainties estimation
^^^^^^^^^^^^^^^^^^^^^^^^

Since the available constrains on the singlet anomalous dimension are not sufficient
to determine their behavior exactly, for instance the poles at :math:`N=1` and :math:`N=0` are not fully known,
we need to account for a possible source of uncertainties arising during the approximation.
This uncertainty is neglected in the non-singlet case.

The procedure is performed in two steps for each different anomalous dimension separately.
First, we solve the system associated to the 4 known moments,
minus the known limits, using different functional bases.
Any possible candidate contains 4 elements and is obtained with the following prescription:

    1. one function is leading small-N unknown contribution, which correspond to the highest power unknown for the pole at :math:`N=1`,

    2. one function is the leading large-N unknown contribution,

    3. the remaining functions are chosen from of a batch of functions describing sub-leading unknown terms both for the small-N and large-N limit.

This way we generate a large set of independent candidates, roughly 70 for each anomalous dimension,
and by taking the standard deviation of the solutions we get as an estimate of the parametrization uncertainties.
When looking at the x-space results we must invert/perform the evolution with each solution
and then compute the statical estimators on the final ensemble.
The "best" result is always taken as the average on all the possible variations.

In the second stage we apply some "post fit" selection criteria to reduce the number of
candidates (to :math:`\approx 20`) selecting the most representative elements and discarding clearly unwanted
solutions. This way we can achieve a smoother result and improve the speed of the calculation.

    * Among the functions selected at point 3 we cherry pick candidates
      containing at least one of the leading sub-leading small-N (poles `N=0,1`)
      or large-N unknown contributions, such that the spread of the reduced ensemble is
      not smaller than the full one.

    * By looking at the x-space line integral, we discard any possible outlier
      that can be generated by numerical cancellations.


The following tables summarize all the considered input functions in the
final reduced sets of candidates.

    .. list-table::  :math:`\gamma_{gg}^{(3)}` parametrization basis
        :align: center

        *   - :math:`f_1(N)`
            - :math:`\frac{S_2(N-2)}{N}`
        *   - :math:`f_2(N)`
            - :math:`\frac{1}{N}`
        *   - :math:`f_3(N)`
            - :math:`\frac{1}{N-1},\ \frac{S_1(N)}{N^2}`
        *   - :math:`f_4(N)`
            - :math:`\frac{1}{N-1},\ \frac{1}{N^4},\ \frac{1}{N^3},\ \frac{1}{N^2},\ \frac{1}{(N+1)^3},\ \frac{1}{(N+1)^2},\ \frac{1}{N+1},\ \frac{1}{N+2},\ \mathcal{M}[(1-x)\ln(1-x)],\ \frac{S_1(N)}{N^2}, \ \mathcal{M}[(1-x)^2\ln(1-x)],`

    .. list-table::  :math:`\gamma_{gq}^{(3)}` parametrization basis
        :align: center

        *   - :math:`f_1(N)`
            - :math:`\frac{S_2(N-2)}{N}`
        *   - :math:`f_2(N)`
            - :math:`\frac{S_1^3(N)}{N}`
        *   - :math:`f_3(N)`
            - :math:`\frac{1}{N-1},\ \frac{1}{N^4}`
        *   - :math:`f_4(N)`
            - :math:`\frac{1}{N-1},\ \frac{1}{N^4},\ \frac{1}{N^3},\ \frac{1}{N^2},\ \frac{1}{N},\ \frac{1}{(N+1)^3},\ \frac{1}{(N+1)^2},\ \frac{1}{N+1},\ \frac{1}{N+2},\ \frac{S_1(N-2)}{N},\ \mathcal{M}[\ln^3(1-x)],\ \mathcal{M}[\ln^2(1-x)], \frac{S_1(N)}{N},\ \frac{S_1^2(N)}{N}`

    Note that this table refers only to the :math:`n_f^0` part where we assume no violation of the scaling with :math:`\gamma_{gg}`
    also for the |NLL| term, to help the convergence. We expect that any possible deviation can be parametrized as a shift in the |NNLL| terms
    and in the |NLL| :math:`n_f^1` which are free to vary independently.

Slightly different choices are performed for :math:`\gamma_{gq}^{(3)}` and :math:`\gamma_{qq,ps}^{(3)}`
where 10 moments are known. In this case we can select a larger number of functions in group 3
and following :cite:`Falcioni:2023luc,Falcioni:2023vqq` we use:

    .. list-table::  :math:`\gamma_{qg}^{(3)}` parametrization basis
        :align: center

        *   - :math:`f_1(N)`
            - :math:`\frac{1}{(N-1)^2}`
        *   - :math:`f_2(N)`
            - :math:`\mathcal{M}[\ln^3(1-x)]`
        *   - :math:`f_3(N)`
            - :math:`\frac{1}{N^4},\ \frac{1}{N^3},\ \frac{1}{N^2},\ \frac{1}{N},\frac{1}{N-1}-\frac{1}{N},\ \mathcal{M}[\ln^2(1-x)]`
        *   - :math:`f_4(N)`
            - :math:`\mathcal{M}[\ln(x) \ln(1-x)],\ \mathcal{M}[\ln(1-x)],\ \mathcal{M}[(1-x)\ln^3(1-x)],\ \mathcal{M}[(1-x)\ln^2(1-x)],\ \mathcal{M}[(1-x)\ln(1-x)],\ \frac{1}{1+N}`

    .. list-table::  :math:`\gamma_{qq,ps}^{(3)}` parametrization basis
        :align: center

        *   - :math:`f_1(N)`
            - :math:`-\frac{1}{(N-1)^2} + \frac{1}{N^2}`
        *   - :math:`f_2(N)`
            - :math:`\mathcal{M}[(1-x)\ln^2(1-x)]`
        *   - :math:`f_{3,\dots,8}(N)`
            - :math:`\frac{1}{N^4},\ \frac{1}{N^3},\ \mathcal{M}[(1-x)\ln(1-x)],\ \mathcal{M}[(1-x)^2\ln^2(1-x)],\ \frac{1}{N-1}-\frac{1}{N},\ \mathcal{M}[(1-x)\ln(x)]`
        *   - :math:`f_{9,10}(N)`
            - :math:`\mathcal{M}[(1-x)(1+2x)],\ \mathcal{M}[(1-x)x^2],\ \mathcal{M}[(1-x) x (1+x)],\ \mathcal{M}[(1-x)]`

Note that for :math:`\gamma_{qq,ps},\gamma_{qg}` the parts proportional
to :math:`n_f^0` are not present.
