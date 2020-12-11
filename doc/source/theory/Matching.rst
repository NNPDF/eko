Matching Conditions on Crossing Thresholds
==========================================

In a |VFNS| the :doc:`evolution distributions <FlavorSpace>` has to be matched across the mass thresholds provided by
the :class:`~eko.thresholds.ThresholdsAtlas` class.
We denote the solution of the :doc:`DGLAP equation <DGLAP>` in :doc:`Mellin space <Mellin>` as

.. math ::
    \tilde{f_j}(Q^2_1)= \tilde E_{jk}(Q^2_1\leftarrow Q^2_0) \tilde{f_k}(Q^2_0)

For the singlet sector (:math:`\Sigma` and :math:`g`), we define the singlet evolution kernel matrix

.. math ::
    \ES{Q_1^2}{Q_0^2} = \begin{pmatrix}
        \tilde E_{qq} & \tilde E_{qg}\\
        \tilde E_{gq} & \tilde E_{gg}
    \end{pmatrix}(Q_1^2\leftarrow Q_0^2)

which is the only coupled system amongst the |DGLAP| equations.

Next, we list the explicit matching conditions for the different evolution distributions up to |LO|.
Note, that the non-trivial matching of the discontinuities only enters at |NNLO|.

Zero Thresholds
---------------

Here, we consider :math:`\mu_{q}^2 < Q_0^2 < Q_1^2 < \mu_{q+1}^2` and we assume that
:math:`\mu_q` is the matching threshold of the :math:`n_f`-th flavor. This configuration corresponds
effectively to a |FFNS|.
All distributions simply evolve with their associated operator.
The singlet sector and the full valence distributions are given by

.. math ::
        \dSV{n_f}{Q_1^2} &= \ES{Q^2_1}{Q_0^2} \dSV{n_f}{Q_0^2}\\
        \dVf{n_f}{Q_1^2} &= \Ensv{Q^2_1}{Q_0^2} \dVf{n_f}{Q_0^2}

If the valence-like/singlet-like non-singlet distributions are already active,
they keep evolving from themselves:

.. math ::
    \dVj{j}{n_f}{Q_1^2} &= \Ensm{Q^2_1}{Q_0^2} \dVj{j}{n_f}{Q_0^2} \\
    \dTj{j}{n_f}{Q_1^2} &= \Ensp{Q^2_1}{Q_0^2} \dTj{j}{n_f}{Q_0^2} \\
     &\text{for }j=3,\ldots, n_f^2-1

Otherwise, they are generated dynamically by the full valence distribution or the singlet
sector respectively:

.. math ::
    \dVj{k}{n_f}{Q_1^2} &= \Ensv{Q^2_1}{Q_0^2} \dVf{n_f}{Q_0^2} \\
    \dTj{k}{n_f}{Q_1^2} &= \left(1, 0\right)\ES{Q_1^2}{Q_0^2}\dSV{n_f}{Q_0^2} \\
     &\text{for }k=(n_f+1)^2-1, \ldots, 35

and making the distributions thus linearly dependent :math:`V_k = V, T_k = \Sigma`
(as they should).

One Threshold
-------------

Here, we consider :math:`\mu_q^2 < Q_0^2 < \mu_{q+1}^2 < Q_1^2 < \mu_{q+2}^2` and we assume that
:math:`\mu_q` is the matching threshold of the :math:`n_f`-th flavor.
The singlet sector and the full valence distributions are given by

.. math ::
    \dSV{n_f+1}{Q_1^2}    &= \ES{Q^2_1}{m_{q+1}^2} \ES{m_{q+1}^2}{Q_0^2} \dSV{n_f}{Q_0^2} \\
    \dVj{j}{n_f+1}{Q_1^2} &= \Ensv{Q^2_1}{m_{q+1}^2} \Ensv{m_{q+1}^2}{Q^2_0} \dVf{n_f}{Q_0^2}

If the valence-like/singlet-like non-singlet distributions have already been active before
the threshold, they keep evolving from themselves

.. math ::
    \dVj{j}{n_f+1}{Q_1^2} &= \Ensm{Q^2_1}{m_{q+1}^2}\Ensm{m_{q+1}^2}{Q_0^2} \dVj{j}{n_f}{Q_0^2}\\
    \dTj{j}{n_f+1}{Q_1^2} &= \Ensp{Q^2_1}{m_{q+1}^2}\Ensp{m_{q+1}^2}{Q_0^2} \dTj{j}{n_f}{Q_0^2}\\
     &\text{for }j=3,\ldots, n_f^2-1

The two distributions which become active after crossing the threshold are generated
dynamically up to the threshold and then set themselves apart:

.. math ::
    \dVj{j'}{n_f+1}{Q_1^2} &= \Ensm{Q^2_1}{m_{q+1}^2}\Ensv{m_{q+1}^2}{Q_0^2} \dVf{n_f}{Q_0^2} \\
    \dTj{j'}{n_f+1}{Q_1^2} &= \Ensp{Q^2_1}{m_{q+1}^2}\left(1,0\right)\ES{m_{q+1}^2}{Q_0^2} \dSV{n_f}{Q_0^2} \\
     & \text{for }j'=(n_f+1)^2-1

The remaining distributions are generated again purely dynamically:

.. math ::
    \dVj{k}{n_f+1}{Q_1^2} &= \Ensv{Q^2_1}{m_{q+1}^2}\Ensv{m_{q+1}^2}{Q_0^2} \dVf{n_f}{Q_0^2} \\
    \dTj{k}{n_f+1}{Q_1^2} &= \left(1, 0\right)\ES{Q_1^2}{m_{q+1}^2}\ES{m_{q+1}^2}{Q_0^2}\dSV{n_f}{Q_0^2} \\
     & \text{for }k=(n_f+2)^2-1, \ldots, 35

Two Thresholds
--------------

Here, we consider :math:`\mu_q^2 < Q_0^2 < \mu_{q+1}^2 < \mu_{q+2}^2 < Q_1^2 < \mu_{q+3}^2` and we assume that
:math:`\mu_q` is the matching threshold of the :math:`n_f`-th r.
The singlet sector and the full valence distributions are given by

.. math ::
    \dSV{n_f+2}{Q_1^2}    &= \ES{Q^2_1}{m_{q+2}^2} \ES{m_{q+2}^2}{m_{q+1}^2} \ES{m_{q+1}^2}{Q_0^2} \dSV{n_f}{Q_0^2} \\
    \dVj{j}{n_f+2}{Q_1^2} &= \Ensv{Q^2_1}{m_{q+2}^2} \Ensv{m_{q+2}^2}{m_{q+1}^2} \Ensv{m_{q+1}^2}{Q^2_0} \dVf{n_f}{Q_0^2}

If the valence-like/singlet-like non-singlet distributions have already been active before
the threshold, they keep evolving from themselves

.. math ::
    \dVj{j}{n_f+2}{Q_1^2} &= \Ensm{Q^2_1}{m_{q+2}^2}\Ensm{m_{q+2}^2}{m_{q+1}^2}\Ensm{m_{q+1}^2}{Q_0^2} \dVj{j}{n_f}{Q_0^2}\\
    \dTj{j}{n_f+2}{Q_1^2} &= \Ensp{Q^2_1}{m_{q+2}^2}\Ensp{m_{q+2}^2}{m_{q+1}^2}\Ensp{m_{q+1}^2}{Q_0^2} \dTj{j}{n_f}{Q_0^2}\\
     &\text{for }j=3,\ldots, n_f^2-1

The two distributions which become active after crossing the *first* threshold are generated
dynamically up to the first threshold and then set themselves apart:

.. math ::
    \dVj{j'}{n_f+2}{Q_1^2} &= \Ensm{Q^2_1}{m_{q+2}^2}\Ensm{m_{q+2}^2}{m_{q+1}^2}\Ensv{m_{q+1}^2}{Q_0^2} \dVf{n_f}{Q_0^2} \\
    \dTj{j'}{n_f+2}{Q_1^2} &= \Ensp{Q^2_1}{m_{q+2}^2}\Ensp{m_{q+2}^2}{m_{q+1}^2}\left(1,0\right)\ES{m_{q+1}^2}{Q_0^2} \dSV{n_f}{Q_0^2} \\
     & \text{for }j'=(n_f+1)^2-1

The two distributions which become active after crossing the *second* threshold are generated
dynamically up to the second threshold and then set themselves apart:

.. math ::
    \dVj{j''}{n_f+2}{Q_1^2} &= \Ensm{Q^2_1}{m_{q+2}^2}\Ensv{m_{q+2}^2}{m_{q+1}^2}\Ensv{m_{q+1}^2}{Q_0^2} \dVf{n_f}{Q_0^2} \\
    \dTj{j''}{n_f+2}{Q_1^2} &= \Ensp{Q^2_1}{m_{q+2}^2}\left(1,0\right)\ES{m_{q+2}^2}{m_{q+1}^2} \ES{m_{q+1}^2}{Q_0^2} \dSV{n_f}{Q_0^2} \\
     & \text{for }j''=(n_f+2)^2-1

If there is a distributions remaining it is generated again purely dynamically:

.. math ::
    \dVj{k}{n_f+2}{Q_1^2} &= \Ensv{Q^2_1}{m_{q+2}^2}\Ensv{m_{q+2}^2}{m_{q+1}^2}\Ensv{m_{q+1}^2}{Q_0^2} \dVf{n_f}{Q_0^2} \\
    \dTj{k}{n_f+2}{Q_1^2} &= \left(1, 0\right)\ES{Q_1^2}{m_{q+2}^2}\ES{m_{q+2}^2}{m_{q+1}^2}\ES{m_{q+1}^2}{Q_0^2}\dSV{n_f}{Q_0^2} \\
     & \text{for }k=(n_f+3)^2-1

Three Thresholds
----------------

Here, we consider :math:`0 < Q_0^2 < \mu_{c}^2 < \mu_{b}^2 < \mu_{t}^2 < Q_1^2 < \infty`.
The singlet sector and the full valence distributions are given by

.. math ::
    \dSV{6}{Q_1^2} &=       \ES{Q^2_1}{m_{t}^2} \ES{m_t^2}{m_{b}^2} \\
                   & \quad  \ES{m_b^2}{m_{c}^2} \ES{m_{c}^2}{Q_0^2} \dSV{3}{Q_0^2} \\
    \dVj{j}{6}{Q_1^2} &=      \Ensv{Q^2_1}{m_{t}^2}   \Ensv{m_{t}^2}{m_{b}^2} \\
                      & \quad \Ensv{m_{b}^2}{m_{c}^2} \Ensv{m_{c}^2}{Q^2_0} \dVf{3}{Q_0^2}

The valence-like/singlet-like non-singlet distributions containing flavors up to strange,
they keep evolving from themselves

.. math ::
    \dVj{j}{6}{Q_1^2} &=      \Ensm{Q^2_1}{m_{t}^2}   \Ensm{m_{t}^2}{m_{b}^2} \\
                      & \quad \Ensm{m_{b}^2}{m_{c}^2} \Ensm{m_{c}^2}{Q_0^2} \dVj{j}{3}{Q_0^2} \\
    \dTj{j}{6}{Q_1^2} &=      \Ensp{Q^2_1}{m_{t}^2}   \Ensp{m_t^2}{m_{qb}^2} \\
                      & \quad \Ensp{m_{b}^2}{m_{c}^2} \Ensp{m_{c}^2}{Q_0^2} \dTj{j}{3}{Q_0^2} \\
     &\text{for }j=3,8

The two distributions containing charm are generated dynamically up to the first threshold
and then set themselves apart:

.. math ::
    \dVj{15}{6}{Q_1^2} &=      \Ensm{Q^2_1}{m_{t}^2}   \Ensm{m_{t}^2}{m_{b}^2} \\
                       & \quad \Ensm{m_{b}^2}{m_{c}^2} \Ensv{m_{c}^2}{Q_0^2} \dVf{3}{Q_0^2} \\
    \dTj{15}{6}{Q_1^2} &=      \Ensp{Q^2_1}{m_{t}^2} \Ensp{m_{t}^2}{m_{b}^2} \\
                       & \quad \Ensp{m_{b}^2}{m_{c}^2} \left(1,0\right)\ES{m_{c}^2}{Q_0^2} \dSV{3}{Q_0^2}

The two distributions containing bottom are generated dynamically up to the second threshold
and then set themselves apart:

.. math ::
    \dVj{24}{6}{Q_1^2} &=      \Ensm{Q^2_1}{m_{t}^2}   \Ensm{m_{t}^2}{m_{b}^2} \\
                       & \quad \Ensv{m_{b}^2}{m_{c}^2} \Ensv{m_{c}^2}{Q_0^2} \dVf{3}{Q_0^2} \\
    \dTj{24}{6}{Q_1^2} &=      \Ensp{Q^2_1}{m_{t}^2} \Ensp{m_{t}^2}{m_{b}^2} \\
                       & \quad \left(1,0\right) \ES{m_{b}^2}{m_{c}^2} \ES{m_{c}^2}{Q_0^2} \dSV{3}{Q_0^2}

The two distributions containing top are generated dynamically up to the third threshold
and then set themselves apart:

.. math ::
    \dVj{35}{6}{Q_1^2} &=      \Ensm{Q^2_1}{m_{t}^2}   \Ensv{m_{t}^2}{m_{b}^2} \\
                       & \quad \Ensv{m_{b}^2}{m_{c}^2} \Ensv{m_{c}^2}{Q_0^2} \dVf{3}{Q_0^2} \\
    \dTj{35}{6}{Q_1^2} &=      \Ensp{Q^2_1}{m_{t}^2} \left(1,0\right) \ES{m_{t}^2}{m_{b}^2} \\
                       & \quad \ES{m_{b}^2}{m_{c}^2} \ES{m_{c}^2}{Q_0^2} \dSV{3}{Q_0^2}

Intrinsic evolution
-------------------

We also consider the evolution of intrinsic heavy |PDF|. Since these are massive partons they can not
split any collinear particles and thus they do not participate in the |DGLAP| evolution. Instead their
evolution is simpliy an indentiy operation: e.g. for an intrinsic distribution we get for
:math:`m_c^2 > Q_1^2 > Q_0^2`:

.. math ::
    \tilde c(Q_1^2) &= \tilde c(Q_0^2)\\
    \tilde {\bar c}(Q_1^2) &= \tilde{\bar c}(Q_0^2)

After crossing the mass threshold (charm in this example) the |PDF| can not be considered intrinsic
any loger. Here they have to be rejoined with their evolution basis elements and take then again
part in the ordinary collinear evolution. This twofold behavior leads in the context of the
FONLL matching scheme :cite:`Forte:2010ta` to considerable simplifications :cite:`Ball:2015dpa`.