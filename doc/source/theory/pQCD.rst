pQCD ingredients
================

Strong Coupling
---------------

Implementation: :class:`~eko.strong_coupling.StrongCoupling`.

We use perturbative QCD with the running coupling
:math:`a_s(\mu_R^2) = \alpha_s(\mu_R^2)/(4\pi)` given at 5-loop by
:cite:`Herzog:2017ohr` :cite:`Luthe:2016ima` :cite:`Baikov:2016tgj`

.. math ::
    \frac{da_s(\mu_R^2)}{d\ln\mu_R^2} = \beta(a_s(\mu_R^2)) \
    = - \sum\limits_{n=0} \beta_n a_s^{n+2}(\mu_R^2)

It is usefull to define in addition :math:`b_k = \beta_k/\beta_0, k>0`.

We implement two different strategies to solve the renormalization group equation (RGE):

- ``method="exact"``: Solve using :func:`scipy.integrate.solve_ivp`.
  In |LO| we fall back to the expanded solution as this is already the true solution.
- ``method="expanded"``: using approximate solutions:

.. math ::
    a^{\text{LO}}_s(\mu_R^2)  &= \frac{a_s(\mu_0^2)}{1 + a_s(\mu_0^2) \beta_0 \ln(\mu_R^2/\mu_0^2)} \\
    a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2) &= a^{\text{LO}}_s(\mu_R^2)-b_1 \left[a^{\text{LO}}_s(\mu_R^2)\right]^2 \ln\left(1+a_s(\mu_0^2) \beta_0 \ln(\mu_R^2/\mu_0^2)\right) \\
    a^{\text{NNLO}}_{s,\text{exp}}(\mu_R^2) &= a^{\text{LO}}_s(\mu_R^2)\left[1 + a^{\text{LO}}_s(\mu_R^2)\left(a^{\text{LO}}_s(\mu_R^2) - a_s(\mu_0^2)\right)(b_2 - b_1^2) \right.\\
                                        & \hspace{60pt} \left. + a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2) b_1 \ln\left(a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2)/a_s(\mu_0^2)\right)\right]

When the renormalization scale crosses a flavor threshold matching conditions have to be
applied :cite:`Schroder:2005hy` :cite:`Chetyrkin:2005ia`.

Splitting Functions
-------------------

The Altarelli-Parisi splitting kernels can be expanded in powers of the strong
coupling :math:`a_s(\mu^2)` and are given by :cite:`Moch:2004pa` :cite:`Vogt:2004mw`

.. math ::
    \mathbf{P}(x,a_s(\mu^2)) &= \sum\limits_{j=0} a_s^{j+1}(\mu^2) \mathbf{P}^{(j)}(x) \\
    {\gamma}^{(j)}(N) &= -\mathcal{M}[\mathbf{P}^{(j)}(x)](N)

Note the additional minus in the definition of :math:`\gamma`.

Scale Variations
----------------

The usual procedure in solving |DGLAP| that is also imployed :doc:`here </theory/DGLAP>` is to rewrite
the equations in term of the running coupling :math:`a_s` assuming the factorization scale
:math:`\mu_F^2` (the inherit scale of the |PDF|) and the renormalization scale :math:`\mu_R^2`
(the inherit scale for the strong coupling) to be equal. This constraint can however be lifted by a
suitable redefinition of the splitting kernels :cite:`Vogt:2004ns`:

.. math ::
    \gamma^{(1)}(N) &\to \gamma^{(1)}(N) - \beta_0 \ln(\mu_F^2/\mu_R^2) \gamma^{(0)} \\
    \gamma^{(2)}(N) &\to \gamma^{(2)}(N) - 2 \beta_0 \ln(\mu_F^2/\mu_R^2) \gamma^{(1)} - ( \beta_1 \ln(\mu_F^2/\mu_R^2) - \beta_0^2 \ln^2(\mu_F^2/\mu_R^2) )  \gamma^{(0)}


while keeping the evalutation of the strong coupling always at :math:`\mu_R^2`.
Estimating the theoretical uncertanties imposed on |PDF| determination due to missing higher
order corrections using scale variation in the evolution corresponds to schemes A and B
in :cite:`AbdulKhalek:2019ihb`.