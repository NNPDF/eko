Matching Conditions on Crossing Thresholds
==========================================

In a |VFNS| one considers several matching thresholds (as provided by the :class:`~eko.thresholds.ThresholdsAtlas`)
where the number of active, light flavors that are participating in the :doc:`DGLAP equation <DGLAP>` changes
by one unit: :math:`n_f \to n_f +1`. This leads to the complication that one the sides of the thresholds the distributions
to not behave in the same matter: in esp. the new quark distributions :math:`q_{n_f+1}(x,\mu_F^2) = h(x,\mu_F^2)` and
:math:`\overline h(x,\mu_F^2)` did not take part in the evolution below the threshold, but above they do.
This mismatch in the evolution is accounted for by the *matching conditions*. We will decompose this operation into two independet
steps: first, the true QCD induced |OME| :math:`\mathbf{M}^{(n_f)}` that are given by perturbative calculations,
and, second, the necessary :doc:`flavor space rotation <FlavorSpace>` :math:`\mathbf{R}^{(n_f)}` to fit the new evolution basis.

We denote the solution of the :doc:`DGLAP equation <DGLAP>` in a region with *no* threshold
(:math:`\mu_{q}^2 < Q_0^2 < Q_1^2 < \mu_{q+1}^2`) in :doc:`Mellin space <Mellin>` as

.. math ::
    \tilde{\mathbf{f}}(Q^2_1)= \tilde{\mathbf{E}}(Q^2_1\leftarrow Q^2_0) \tilde{\mathbf{f}}(Q^2_0)

The bold font indicates the vector space spanned by the flavor space and the equation decouples mostly in the evolution basis.

We can then denote the solution with a single threshold (:math:`\mu_h^2 < Q_0^2 < \mu_{h+1}^2 < Q_1^2 < \mu_{h+2}^2`) as

.. math ::
    \tilde{\mathbf{f}}(Q^2_1)= \tilde{\mathbf{E}}(Q^2_1\leftarrow \mu_{h+1}^2) {\mathbf{R}^{(n_f)}} {\mathbf{M}^{(n_f)}}(\mu_{h+1}^2) \tilde{\mathbf{E}}(\mu_{h+1}^2\leftarrow Q^2_0) \tilde{\mathbf{f}}(Q^2_0)

In case more than one threshold scale is present the matching procedure is iterared on all diffrent scales starting form
the lowest one.

Basis rotation
--------------

The rotation matrices :math:`\mathbf{R}^{(n_f)}` are given by

.. math ::
    \dSVe{n_f+1}{\mu_{h}^2} &= {\mathbf{R}}_S^{(n_f)} \dSVi{n_f+1}{\mu_{h}^2} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & - n_f \end{pmatrix} \dSVi{n_f+1}{\mu_{h}^2} \\
    \dVe{n_f+1}{\mu_{h}^2} &= {\mathbf{R}}_{ns}^{(n_f)} \dVi{n_f+1}{\mu_{h}^2} = \begin{pmatrix} 1 & 1 \\ 1 & - n_f \end{pmatrix} \dVi{n_f+1}{\mu_{h}^2} \\
    & \text{for }j=(n_f+1)^2-1\\
    {\mathbf{R}}_{ns\pm}^{(n_f)} &= \mathbf 1 ~ \text{otherwise}


Pertubative Operator Matrix Element
-----------------------------------

.. math ::
    \dSVi{n_f+1}{\mu_{h}^2} &= \mathbf{M}_{S}(\mu_{h}^2) \dSVi{n_f}{\mu_{h}^2} \\
    \dVi{n_f+1}{\mu_{h}^2} &= \mathbf{M}_{ns}(\mu_{q+1}^2) \dVi{n_f}{\mu_{h}^2}


with :math:`\mathbf{M}` being the matching matrices computed order by order in :math:`a_s`: 

.. math ::
    \mathbf{M}_{X}(\mu_{h}^2) &= \mathbf{I} + a_s(\mu_{h}^2)  \mathbf{A}_{X}^{(1)} + a_s^2(\mu_{h}^2) \mathbf{A}_{X}^{(2)} \\
    & \text{for } X=S,ns \\


and :math:`\mathbf{A}_{X}^{i}` the operator matrix elements (|OME|), given by the following:

.. math ::
    \mathbf{A}_{ns}^{(1)} &= \begin{pmatrix} 0 & 0 \\ 0 & A_{HH}^{(1)}\end{pmatrix} \\
    \mathbf{A}_{S}^{(1)} &= \begin{pmatrix} A_{gg,H}^{s,(1)} & 0 & A_{gH}^{s,(1)} \\ 0 & 0 & 0 \\ A_{Hg}^{s,(1)} & 0 & A_{HH}^{(1)} \end{pmatrix} \\
    \mathbf{A}_{ns}^{(2)} &= \begin{pmatrix} A_{qq,H}^{ns,(2)} & 0 \\ A_{Hq}^{ps,(2)} & 0 \end{pmatrix} \\
    \mathbf{A}_{S}^{(2)} &= \begin{pmatrix} A_{gg,H}^{s,(2)} & A_{gq,H}^{s,(2)} & 0 \\ 0 & A_{qq,H}^{ns,(2)} & 0 \\ A_{Hg}^{s,(2)} & A_{Hq}^{ps,(2)} & 0 \end{pmatrix} \\


The coefficients :math:`A^{x}_{i}(z,\mu_{h}^2)` have been firstly computed in :cite:`Buza_1998` and have been Mellin tranformed to be used inside EKO. They depends on the scale :math:`\mu_{h}^2` only through the logaritm :math:`ln(\frac{\mu_{q}^2}{m_{q}^2})`,
in particular the coefficient :math:`A_{gg,H}^{s,(1)}` is fully proprtional to :math:`ln(\frac{\mu_{h}^2}{m_{h}^2})`. 

We remark that contributions of the higher quark at |NNLO| have not been computed yet, thus the elements :math:`A_{qH}^{(2)},A_{gH}^{(2)}A_{HH}^{(2)}` are not encoded in EKO despite of being present.
On the other hand the elements :math:`A_{qq}^{ps},A_{qg}` are known to start at order :math:`O(a_s^3)`.


The other valence-like/singlet-like non-singlet distributions that were already active before the threshold, continue to evolve from themselves
under the condition:

.. math ::
    \dVj{j}{n_f+1}{\mu_h^2} &= M_{ns}(m_{h}^2) \dVj{j}{n_f}{\mu_h^2}\\
    \dTj{j}{n_f+1}{\mu_h^2} &= M_{ns}(m_{h}^2) \dTj{j}{n_f}{\mu_h^2}\\
    &\text{for }j=3,\ldots, n_f^2-1

Intrinsic evolution
-------------------

We also consider the evolution of intrinsic heavy |PDF|. Since these are massive partons they can not
split any collinear particles and thus they do not participate in the |DGLAP| evolution. Instead, their
evolution is simpliy an indentiy operation: e.g. for an intrinsic distribution we get for
:math:`m_c^2 > Q_1^2 > Q_0^2`:

.. math ::
    \tilde c(Q_1^2) &= \tilde c(Q_0^2)\\
    \tilde {\bar c}(Q_1^2) &= \tilde{\bar c}(Q_0^2)

After crossing the mass threshold (charm in this example) the |PDF| can not be considered intrinsic
any longer. Here, they have to be rejoined with their evolution basis elements and take then again
part in the ordinary collinear evolution. This twofold behavior leads in the context of the
FONLL matching scheme :cite:`Forte:2010ta` to considerable simplifications :cite:`Ball:2015dpa`.

Backward evolution
------------------

When looking at the backward evolution and passing the threshold :math:`\mu_{h}^2` the PDFs in the higher patch are rotated in to the flavor basis
before the matching with:


and then matched to the PDFs in the lower patch with the inverse of :math:`\mathbf{M}`. 
EKO implements two different strategies to perform this operation, that can be specied with the parameter ``backward_inversion``:

- ``backward_inversion = 'exact'``: the matching matrices are inverted exactly in N space, and then integrted element by element
- ``backward_inversion = 'expanded'``: the matching matrices are inverted through a pertubative exapnsion in :math:`a_s` before the Mellin inversion:

.. math ::
    \mathbf{M}_{X,exp}^{-1}(\mu_{q}^2) &= \mathbf{I} - a_s(\mu_{q}^2)  \mathbf{A}_{X}^{(1)} + a_s^2(\mu_{q}^2) \left [ \mathbf{A}_{X}^{(2)} -  {\mathbf{A}_{X}^{(1)}}^2 \right ] + o(a_s^3) \\

We emphasize that in the backward evolution, below the threshold, the remaining high quark PDFs are always intrinsic and do not evolve anymore.
