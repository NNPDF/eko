Matching Conditions on Crossing Thresholds
==========================================

In a |VFNS| one considers several matching thresholds (as provided by the :class:`~eko.thresholds.ThresholdsAtlas`)
where the number of active, light flavors that are participating in the :doc:`DGLAP equation <DGLAP>` changes
by one unit: :math:`n_f \to n_f +1`. This means the distributions do not behave in the same matter above and below
the threhsold: in esp. the new quark distributions :math:`q_{n_f+1}(x,\mu_F^2) = h(x,\mu_F^2)` and
:math:`\overline h(x,\mu_F^2)` did not take part in the evolution below the threshold, but above they do.
This mismatch in the evolution is accounted for by the *matching conditions*.

In the following we will denote the number of active flavors by a supscript :math:`{}^{(n_f)}`.
We denote the solution of the :doc:`DGLAP equation <DGLAP>` in a region with a *fixed* number of active flavors, i.e. *no* threshold
present :math:`\left(\mu_{h}^2 < Q_0^2 < Q_1^2 < \mu_{h+1}^2\right)`, in :doc:`Mellin space <Mellin>` as

.. math ::
    \tilde{\mathbf{f}}^{(n_f)}(Q^2_1)= \tilde{\mathbf{E}}^{(n_f)}(Q^2_1\leftarrow Q^2_0) \tilde{\mathbf{f}}^{(n_f)}(Q^2_0)

The bold font indicates the vector space spanned by the :doc:`flavor space <FlavorSpace>` and the equations decouple mostly
in the :ref:`Intrinsic Evolution Basis <theory/FlavorSpace:Intrinsic Evolution Bases>`.

If a single threshold :math:`\left(\mu_{h-1}^2 < Q_0^2 < \mu_{h}^2 < Q_1^2 < \mu_{h+1}^2\right)` is present
we decompose the matching into two independet steps:
first, the true QCD induced |OME| :math:`\mathbf{A}^{(n_f)}(\mu_{h}^2)` that are given by perturbative calculations,
and, second, the necessary :doc:`flavor space rotation <FlavorSpace>` :math:`\mathbf{R}^{(n_f)}` to fit the
new :ref:`Intrinsic Evolution Basis <theory/FlavorSpace:Intrinsic Evolution Bases>`.
We can then denote the solution as

.. math ::
    \tilde{\mathbf{f}}^{(n_f+1)}(Q^2_1)= \tilde{\mathbf{E}}^{(n_f+1)}(Q^2_1\leftarrow \mu_{h}^2) {\mathbf{R}^{(n_f)}} \tilde{\mathbf{A}}^{(n_f)}(\mu_{h}^2) \tilde{\mathbf{E}}^{(n_f)}(\mu_{h}^2\leftarrow Q^2_0) \tilde{\mathbf{f}}^{(n_f)}(Q^2_0)

In the case of more than one threshold beeing present, the matching procedure is iterated on all thresholds.


Operator Matrix Elements
------------------------

The matching matrices :math:`\mathbf{A}^{(n_f)}(\mu_{h+1}^2)` mediate between :math:`\mathcal F_{iev,n_f}^{(n_f)}`
and :math:`\mathcal F_{iev,n_f}^{(n_f+1)}`, i.e. they transform the basis vectors of the :math:`n_f`-flavors space
in a :math:`n_f`-flavor scheme to the :math:`(n_f+1)`-flavor scheme. Hence, the supscript refers to the flavor scheme
with a smaller number of active flavors. To compute the matrices in a minimal coupled system we decompose the
:ref:`Intrinsic Evolution Basis <theory/FlavorSpace:Intrinsic Evolution Bases>` :math:`\mathcal F_{iev,n_f}` into
several subspaces (of course irrespective of the |FNS|):

.. math ::
    \mathcal F_{iev,3,S,c^+} &= \span(g,\Sigma,c^+)\\
    \mathcal F_{iev,3,nsv,c^-} &= \span(V,c^-)\\
    \mathcal F_{iev,3,ns+} &= \span(T_3,T_8)\\
    \mathcal F_{iev,3,ns-} &= \span(V_3,V_8)\\
    \mathcal F_{iev,3,hh} &= \span(b^+,b^-,t^+,t^-)\\
    \mathcal F_{iev,n_f} &= \mathcal F_{iev,3,S,c^+} \otimes \mathcal F_{iev,3,nsv,c^-} \otimes \mathcal F_{iev,3,ns+}
                            \otimes \mathcal F_{iev,3,ns-} \otimes \mathcal F_{iev,3,hh}

We can then write the matching matrices :math:`\mathbf{A}^{(n_f)}(\mu_{h+1}^2)` as

.. math ::
    \dSVip{n_f}{\mu_{h}^2} &= \tilde{\mathbf{A}}_{S,h^+}^{(n_f)}(\mu_{h}^2) \dSVi{n_f}{\mu_{h}^2} \\
    \dVip{n_f}{\mu_{h}^2} &= \tilde{\mathbf{A}}_{nsv,h^-}^{(n_f)}(\mu_{h}^2) \dVi{n_f}{\mu_{h}^2} \\
    \dVj{j}{n_f+1}{\mu_h^2} &= \tilde{A}_{ns-}^{(n_f)}(\mu_{h}^2) \dVj{j}{n_f}{\mu_h^2}\\
    \dTj{j}{n_f+1}{\mu_h^2} &= \tilde{A}_{ns+}^{(n_f)}(\mu_{h}^2) \dTj{j}{n_f}{\mu_h^2}\\
    &\text{for }j=3,\ldots, n_f^2-1


where :math:`\mathbf{A}^{(n_f)}(\mu_{h+1}^2)` can be computed order by order in :math:`a_s`:

.. math ::
    \mathbf{A}^{(n_f)}(\mu_{h}^2) = \mathbf{I} + a_s^{(n_f)}(\mu_{h}^2)  \mathbf{A}^{(n_f),(1)} + \left(a_s^{(n_f)}(\mu_{h}^2)\right)^2 \mathbf{A}^{(n_f),(2)}


and the :math:`\mathbf{A}^{(n_f),(k)}` are given upto |NNLO| by the following expressions:

.. math ::
    \mathbf{A}_{S,h^+}^{(n_f),(1)} &= \begin{pmatrix} A_{gg,H}^{S,(1)} & 0 & A_{gH}^{S,(1)} \\ 0 & 0 & 0 \\ A_{Hg}^{S,(1)} & 0 & A_{HH}^{(1)} \end{pmatrix} \\
    \mathbf{A}_{nsv,h^-}^{(n_f),(1)} &= \begin{pmatrix} 0 & 0 \\ 0 & A_{HH}^{(1)}\end{pmatrix} \\
    \mathbf{A}_{S,h^+}^{(n_f),(2)} &= \begin{pmatrix} A_{gg,H}^{S,(2)} & A_{gq,H}^{S,(2)} & 0 \\ 0 & A_{qq,H}^{ns,(2)} & 0 \\ A_{Hg}^{S,(2)} & A_{Hq}^{ps,(2)} & 0 \end{pmatrix} \\
    \mathbf{A}_{nsv,h^-}^{(n_f),(2)} &= \begin{pmatrix} A_{qq,H}^{ns,(2)} & 0 \\ 0 & 0 \end{pmatrix}


The coefficients :math:`A^{(n_f),(k)}_{ij}(z,\mu_{h}^2)` have been firstly computed in :cite:`Buza_1998` and have
been :doc:`Mellin tranformed </theory/Mellin>` to be used inside EKO.
They depend on the scale :math:`\mu_{h}^2` is only through the logarithm :math:`\ln(\mu_{h}^2/m_{h}^2)`,
in particular the coefficient :math:`A_{gg,H}^{S,(1)}` is fully proprtional to :math:`\ln(\mu_{h}^2/m_{h}^2)`.

We remark that contributions of the higher quark at |NNLO| have not been computed yet, thus the elements :math:`A_{qH}^{(2)},A_{gH}^{(2)}A_{HH}^{(2)}` are not encoded in EKO despite of being present.
On the other hand the elements :math:`A_{qq}^{ps},A_{qg}` are known to start at |N3LO|.

The |OME| are also required in the context of the FONLL matching scheme :cite:`Forte:2010ta`.
For :ref:`Intrinsic Evolution <theory/DGLAP:Intrinsic Evolution>` this leads to considerable simplifications :cite:`Ball:2015dpa`.

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

- ``backward_inversion = 'exact'``: the matching matrices are inverted exactly in N space, and then integrated element by element
- ``backward_inversion = 'expanded'``: the matching matrices are inverted through a pertubative expansion in :math:`a_s` before the Mellin inversion:

.. math ::
    \mathbf{A}_{exp}^{-1}(\mu_{h}^2) &= \mathbf{I} - a_s(\mu_{h}^2)  \mathbf{A}^{(1)} + a_s^2(\mu_{h}^2) \left [ \mathbf{A}^{(2)} -  \left(\mathbf{A}^{(1)}\right)^2 \right ] + O(a_s^3) \\

We emphasize that in the backward evolution, below the threshold, the remaining high quark PDFs are always intrinsic and do not evolve anymore.
