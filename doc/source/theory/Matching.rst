Crossing Thresholds
===================

In a |VFNS| one considers several matching scales
where the number of active, light flavors that are participating in the :doc:`DGLAP equation <DGLAP>` changes
by one unit: :math:`n_f \to n_f +1`. This means the distributions do not behave in the same matter above and below
the threshold: especially the new quark distributions :math:`q_{n_f+1}(x,\mu_F^2) = h(x,\mu_F^2)` and
:math:`\overline h(x,\mu_F^2)` did not take part in the evolution below the threshold, but above they do.
This mismatch in the evolution is accounted for by the *matching conditions*.

In the following we will denote the number of active flavors by a supscript :math:`{}^{(n_f)}`.
We denote the solution of the :doc:`DGLAP equation <DGLAP>` in a region with a *fixed* number of active flavors, i.e. *no* threshold
present :math:`\left(\mu_{h}^2 < Q_0^2 < Q_1^2 < \mu_{h+1}^2\right)`, in :doc:`Mellin space <Mellin>` as

.. math ::
    \tilde{\mathbf{f}}^{(n_f)}(Q^2_1)= \tilde{\mathbf{E}}^{(n_f)}(Q^2_1\leftarrow Q^2_0) \tilde{\mathbf{f}}^{(n_f)}(Q^2_0)

The bold font indicates the vector space spanned by the :doc:`flavor space <FlavorSpace>` and the equations decouple mostly
in the :ref:`Intrinsic Evolution Basis <theory/FlavorSpace:Intrinsic QCD Evolution Bases>`.

If a single threshold :math:`\left(\mu_{h-1}^2 < Q_0^2 < \mu_{h}^2 < Q_1^2 < \mu_{h+1}^2\right)` is present
we decompose the matching into two independent steps:
first, the true |QCD| induced |OME| :math:`\mathbf{A}^{(n_f)}(\mu_{h}^2)` that are given by perturbative calculations and expressed in the flavor space,
and, second, the necessary :doc:`flavor space rotation <FlavorSpace>` :math:`\mathbf{R}^{(n_f)}` to fit the
new :ref:`Intrinsic Evolution Basis <theory/FlavorSpace:Intrinsic QCD Evolution Bases>`.
We can then denote the solution as

.. math ::
    \tilde{\mathbf{f}}^{(n_f+1)}(Q^2_1)= \tilde{\mathbf{E}}^{(n_f+1)}(Q^2_1\leftarrow \mu_{h}^2) {\mathbf{R}^{(n_f)}} \tilde{\mathbf{A}}^{(n_f)}(\mu_{h}^2) \tilde{\mathbf{E}}^{(n_f)}(\mu_{h}^2\leftarrow Q^2_0) \tilde{\mathbf{f}}^{(n_f)}(Q^2_0)

In the case of more than one threshold being present, the matching procedure is iterated on all matching scales starting from the lowest one.


Evolution Points
----------------

The matching procedure implies that any target scale :math:`Q` at the associated number of active flavors :math:`n_f` are two
*independent* variables, which, however, are both required to uniquely identify where the evolution is happening.
We thus define the concept of an **Evolution Point**, which is a tuple of a scale and a number of flavors, e.g.

.. math::
    e_1 = (Q_1, n_{f,1})

The concept of evolution points applies to all perturbative |QCD| objects and specifically also for |PDF| .
A more detailed explanation can be found in :cite:`Barontini:2024xgu`.

Often an *implicit definition* for the number of flavors :math:`n_f` is assumed: often it is assumed,
that the number of flavors is given by the number of crossed heavy flavor threshold.
For example if a scale is between the charm and bottom threshold, :math:`\mu_c < Q < \mu_b`,
:math:`n_f=3` is implied (and similar for other cases).
EKO does not make this assumptions, but expects an explicit definition.


Operator Matrix Elements
------------------------

The matching matrices :math:`\mathbf{A}^{(n_f)}(\mu_{h+1}^2)` mediate between :math:`\mathcal F_{iev,n_f}^{(n_f)}`
and :math:`\mathcal F_{iev,n_f}^{(n_f+1)}`, i.e. they transform the basis vectors of the :math:`n_f`-flavors space
in a :math:`n_f`-flavor scheme to the :math:`(n_f+1)`-flavor scheme. Hence, the supscript refers to the flavor scheme
with a smaller number of active flavors. To compute the matrices in a minimal coupled system we decompose the
:ref:`Intrinsic Evolution Basis <theory/FlavorSpace:Intrinsic QCD Evolution Bases>` :math:`\mathcal F_{iev,n_f}` into
several subspaces (below for the example of `n_f = 3`):

.. math ::
    \mathcal F_{iev,3,S,c^+} &= \span(g,\Sigma,c^+)\\
    \mathcal F_{iev,3,nsv,c^-} &= \span(V,c^-)\\
    \mathcal F_{iev,3,ns+} &= \span(T_3,T_8)\\
    \mathcal F_{iev,3,ns-} &= \span(V_3,V_8)\\
    \mathcal F_{iev,3,h} &= \span(b^+,b^-,t^+,t^-)\\
    \mathcal F_{iev,n_f} &= \mathcal F_{iev,3,S,c^+} \otimes \mathcal F_{iev,3,nsv,c^-} \otimes \mathcal F_{iev,3,ns+}
                            \otimes \mathcal F_{iev,3,ns-} \otimes \mathcal F_{iev,3,h}

We can then write the matching matrices :math:`\mathbf{A}^{(n_f)}(\mu_{h+1}^2)` as

.. math ::
    \dSVip{n_f}{\mu_{h}^2} &= \tilde{\mathbf{A}}_{S,h^+}^{(n_f)}(\mu_{h}^2) \dSVi{n_f}{\mu_{h}^2} \\
    \dVip{n_f}{\mu_{h}^2} &= \tilde{\mathbf{A}}_{nsv,h^-}^{(n_f)}(\mu_{h}^2) \dVi{n_f}{\mu_{h}^2} \\
    \dVj{j}{n_f+1}{\mu_h^2} &= \tilde{A}_{ns-}^{(n_f)}(\mu_{h}^2) \dVj{j}{n_f}{\mu_h^2}\\
    \dTj{j}{n_f+1}{\mu_h^2} &= \tilde{A}_{ns+}^{(n_f)}(\mu_{h}^2) \dTj{j}{n_f}{\mu_h^2}\\
    &\text{for }j=3,\ldots, n_f^2-1

Note that in the left hand side basis the distributions :math:`\tilde \Sigma_{(n_f)}, \tilde V_{(n_f)}`
are no longer the ordinary singlet and valence distribution as they
do not contain the new flavor :math:`\tilde h^{+}, \tilde h^{-}`.
Furthermore, in the right side basis :math:`\tilde h^{+}, \tilde h^{-}` are intrinsic contributions.

The :math:`\mathbf{A}^{(n_f)}(\mu_{h+1}^2)` can be computed order by order in :math:`a_s`:

.. math ::
    \mathbf{A}^{(n_f)}(\mu_{h}^2) = \mathbf{I} + \sum_{k=1} \left(a_s^{(n_f+1)}(\mu_{h}^2)\right)^k \mathbf{A}^{(n_f),(k)}


where the :math:`\mathbf{A}^{(n_f),(k)}` are given up to |N3LO| by the following expressions:

.. math ::
    \mathbf{A}_{S,h^+}^{(n_f),(1)} &= \begin{pmatrix} A_{gg,H}^{S,(1)} & 0 & A_{gH}^{S,(1)} \\ 0 & 0 & 0 \\ A_{Hg}^{S,(1)} & 0 & A_{HH}^{(1)} \end{pmatrix} \\
    \mathbf{A}_{nsv,h^-}^{(n_f),(1)} &= \begin{pmatrix} 0 & 0 \\ 0 & A_{HH}^{(1)}\end{pmatrix} \\
    \mathbf{A}_{S,h^+}^{(n_f),(2)} &= \begin{pmatrix} A_{gg,H}^{S,(2)} & A_{gq,H}^{S,(2)} & 0 \\ 0 & A_{qq,H}^{ns,(2)} & 0 \\ A_{Hg}^{S,(2)} & A_{Hq}^{ps,(2)} & 0 \end{pmatrix} \\
    \mathbf{A}_{nsv,h^-}^{(n_f),(2)} &= \begin{pmatrix} A_{qq,H}^{ns,(2)} & 0 \\ 0 & 0 \end{pmatrix} \\
    \mathbf{A}_{S,h^+}^{(n_f),(3)} &= \begin{pmatrix} A_{gg,H}^{S,(3)} & A_{gq,H}^{S,(3)} & 0 \\ A_{qg,H}^{S,(3)} & A_{qq,H}^{ns,(3)} + A_{qq,H}^{ps,(3)} & 0 \\ A_{Hg}^{S,(3)} & A_{Hq}^{ps,(3)} & 0 \end{pmatrix} \\
    \mathbf{A}_{nsv,h^-}^{(n_f),(3)} &= \begin{pmatrix} A_{qq,H}^{ns,(3)} & 0 \\ 0 & 0 \end{pmatrix}

The coefficients :math:`A^{(n_f),(k)}_{ij}(z,\mu_{h}^2)` have been firstly computed in :cite:`Buza_1998` and have
been :doc:`Mellin transformed </theory/Mellin>` to be used inside EKO.
They depend on the scale :math:`\mu_{h}^2` only through the logarithm :math:`\ln(\mu_{h}^2/m_{h}^2)`,
in particular the coefficient :math:`A_{gg,H}^{S,(1)}` is fully proportional to :math:`\ln(\mu_{h}^2/m_{h}^2)`.
During the matching we use :math:`a_s^{(n_f+1)}`: in fact the :math:`a_s` decoupling gives raise to some additional logarithms
:math:`\ln(\mu_{h}^2/m_{h}^2)`, which are cancelled by the OME's :math:`A_{kl,H}`.

|N3LO| matrix elements have been presented in :cite:`Bierenbaum:2009mv` and following publications
:cite:`Ablinger:2010ty,Ablinger:2014vwa,Ablinger:2014uka,Behring:2014eya,Blumlein:2017wxd,Ablinger_2014,Ablinger_2015,Ablinger:2022wbb,Ablinger:2024xtt`.
Parts proportional to :math:`\ln(\mu_{h}^2/m_{h}^2)` are also included up to |N3LO|.

All the contributions are now known analytically. Due to the lengthy and complex expressions
some parts of :math:`A_{Hg}^{S,(3)},A_{Hq}^{S,(3)},A_{gg}^{S,(3)},A_{qq}^{NS,(3)}` have been parameterized.

We remark that contributions of the heavy quark initiated diagrams at |NNLO| and |N3LO| have not been computed yet,
thus the elements :math:`A_{qH}^{(2)},A_{gH}^{(2)}A_{HH}^{(2)}` are not encoded in EKO despite of being present.
On the other hand the elements :math:`A_{qq,H}^{ps},A_{qg,H}` are known to start at |N3LO|.

Additional contributions due to |MSbar| masses are included only up to |NNLO|.

The |OME| are also required in the context of the FONLL matching scheme :cite:`Forte:2010ta`.
For :ref:`Intrinsic Evolution <theory/DGLAP:Intrinsic Evolution>` this leads to considerable simplifications :cite:`Ball:2015dpa`.

Matching conditions for polarized and time-like evolution follow a similar structure. The former being implemented up to
|NNLO| from :cite:`Bierenbaum:2022biv` and the latter up to |NLO| :cite:`Cacciari:2005ry` as the |NNLO| contributions are
currently unknown.

Basis rotation
--------------

The rotation matrices :math:`\mathbf{R}^{(n_f)}` mediate between :math:`\mathcal F_{iev,n_f}^{(n_f+1)}` and :math:`\mathcal F_{iev,n_f+1}^{(n_f+1)}`,
i.e. in the input and output the distributions are already in a scheme with :math:`(n_f+1)`-flavors and the new heavy quark is already non-trivial,
but the basis vectors are still expressed with the elements of the :math:`n_f`-flavors space. The matrices are fixed algebraic quantities and do not
encode perturbative calculations.

The matrices are given by

.. math ::
    \dSVe{n_f+1}{\mu_{h}^2} &= {\mathbf{R}}_{S,h^+}^{(n_f)} \dSVi{n_f+1}{\mu_{h}^2} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & - n_f \end{pmatrix} \dSVi{n_f+1}{\mu_{h}^2} \\
    \dVe{n_f+1}{\mu_{h}^2} &= {\mathbf{R}}_{nsv,h^-}^{(n_f)} \dVi{n_f+1}{\mu_{h}^2} = \begin{pmatrix} 1 & 1 \\ 1 & - n_f \end{pmatrix} \dVi{n_f+1}{\mu_{h}^2} \\
    & \text{for }j=(n_f+1)^2-1\\
    {\mathbf{R}}^{(n_f)} &= \mathbf 1 ~ \text{otherwise}

Backward evolution
------------------

For backward evolution the matching procedure has to be applied in the reversed order: while the inversion of the basis rotation
matrices :math:`\mathbf{R}^{(n_f)}` are easy to invert, this does not apply to the |OME| :math:`\mathbf{A}^{(n_f)}`.
EKO implements two different strategies to perform this operation, that can be specified with the parameter ``backward_inversion``:

- ``backward_inversion = 'exact'``: the matching matrices are inverted exactly in N space, and then integrated entry by entry
- ``backward_inversion = 'expanded'``: the matching matrices are inverted through a perturbative expansion in :math:`a_s` before the Mellin inversion:

.. math ::
    \mathbf{A}_{exp}^{-1}(\mu_{h}^2) &= \mathbf{I} \\
    & - a_s(\mu_{h}^2) \mathbf{A}^{(1)} \\
    & + a_s^2(\mu_{h}^2) \left [ - \mathbf{A}^{(2)} + \left(\mathbf{A}^{(1)}\right)^2 \right ] \\
    & + a_s^3(\mu_{h}^2) \left [ - \mathbf{A}^{(3)} + \mathbf{A}^{(1)} \mathbf{A}^{(2)} + \mathbf{A}^{(2)} \mathbf{A}^{(1)} - \left( \mathbf{A}^{(1)} \right )^3 \right ] \\

We emphasize that in the backward evolution, below the threshold, the remaining high quark PDFs are always intrinsic and do not evolve anymore.
In fact, if the initial PDFs (above threshold) do contain an intrinsic contribution, this has to be evolved below the threshold otherwise momentum sum rules
can be violated.

QED Matching
------------

In the QED case the matching is changed only because of the change of the evolution basis, therefore the only different part will be the basis rotation.
In fact, the |OME| :math:`\mathbf{A}^{(n_f)}(\mu_{h}^2)` don't have |QED| corrections. The matching of the singlet sector is unchanged since it
remains the same with respect to the |QCD| case. The same happens for the matching of the valence component. All the elements :math:`V_i` and :math:`T_i`
are non-singlet components, therefore they are matched with :math:`A_{ns}`. In the end, the new components :math:`\Sigma_\Delta` and :math:`V_\Delta` are matched
with :math:`A_{ns}` since they are both non-singlets.

QED basis rotation
------------------

For the basis rotation we have to consider that we are using the intrinsic unified evolution basis. Here it will be discussed only the rotation to be applied
to the sector :math:`(\Sigma, \Sigma_\Delta, T_i)`, being the rotation of the sector :math:`(V, V_\Delta, V_i)` completely equivalent.
The rotation matrix is given by:

.. math ::
    \begin{pmatrix} \Sigma_{(n_f)} \\ \Sigma_{\Delta,(n_f)} \\ T_{i,(nf)} \end{pmatrix}^{(n_f+1)} =
    \begin{pmatrix} 1 & 0 & 1 \\ a(n_f) & b(n_f) & c(n_f) \\ d(n_f) & e(n_f) & f(n_f) \end{pmatrix}
    \begin{pmatrix} \Sigma_{(n_f)} \\ \Sigma_{\Delta,(n_f)} \\ h^+ \end{pmatrix}^{(n_f)}

where

.. math ::
    a(n_f) & = \frac{1}{n_f}\Bigl(\frac{n_d(n_f+1)}{n_u(n_f+1)}n_u(n_f)-n_d(n_f)\Bigr) \\
    b(n_f) & = \frac{n_f+1}{n_u(n_f+1)}\frac{n_u(n_f)}{n_f} \\
    c(n_f) & = \begin{cases} \frac{n_d(n_f+1)}{n_u(n_f+1)} \quad \text{if $h$ is up-like}\\-1  \quad \text{if $h$ is down-like}\end{cases} \\
    d(n_f) & = \begin{cases} &\frac{n_u(n_f)}{n_f} \quad \text{if $h$ is up-like ($n_f$=3,5)} \\ &\frac{n_d(n_f)}{n_f} \quad \text{if $h$ is down-like ($n_f$=2,4)} \end{cases} \\
    e(n_f) & = \begin{cases} &\frac{n_u(n_f)}{n_f} \quad \text{if $h$ is up-like} \\ &-\frac{n_u(n_f)}{n_f} \quad \text{if $h$ is down-like} \end{cases} \\
    f(n_f) & = \begin{cases} &-1\quad \text{if $h$ is $s$, $c$  ($n_f$=2,3)} \\ &-2 \quad \text{if $h$ is $b$, $t$  ($n_f$=4,5)} \end{cases}
