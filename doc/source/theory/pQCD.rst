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

It is useful to define in addition :math:`b_k = \beta_k/\beta_0, k>0`.

We implement two different strategies to solve the |RGE|:

- ``method="exact"``: Solve using :func:`scipy.integrate.solve_ivp`.
  In |LO| we fall back to the expanded solution as this is already the true solution.
- ``method="expanded"``: using approximate solutions:

.. math ::
    a^{\text{LO}}_s(\mu_R^2) &= \frac{a_s(\mu_0^2)}{1 + a_s(\mu_0^2) \beta_0 \ln(\mu_R^2/\mu_0^2)} \\
    a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2) &= a^{\text{LO}}_s(\mu_R^2)-b_1 \left[a^{\text{LO}}_s(\mu_R^2)\right]^2 \ln\left(1+a_s(\mu_0^2) \beta_0 \ln(\mu_R^2/\mu_0^2)\right) \\
    a^{\text{NNLO}}_{s,\text{exp}}(\mu_R^2) &= a^{\text{LO}}_s(\mu_R^2)\left[1 + a^{\text{LO}}_s(\mu_R^2)\left(a^{\text{LO}}_s(\mu_R^2) - a_s(\mu_0^2)\right)(b_2 - b_1^2) \right.\\
                                        & \hspace{60pt} \left. + a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2) b_1 \ln\left(a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2)/a_s(\mu_0^2)\right)\right]

When the renormalization scale crosses a flavor threshold matching conditions
have to be applied :cite:`Schroder:2005hy,Chetyrkin:2005ia`.

Splitting Functions
-------------------

The Altarelli-Parisi splitting kernels can be expanded in powers of the strong
coupling :math:`a_s(\mu^2)` and are given by :cite:`Moch:2004pa,Vogt:2004mw`

.. math ::
    \mathbf{P}(x,a_s(\mu^2)) &= \sum\limits_{j=0} a_s^{j+1}(\mu^2) \mathbf{P}^{(j)}(x) \\
    {\gamma}^{(j)}(N) &= -\mathcal{M}[\mathbf{P}^{(j)}(x)](N)

Note the additional minus in the definition of :math:`\gamma`.

Scale Variations
----------------

The usual procedure in solving |DGLAP| that is also applied :doc:`here
</theory/DGLAP>` is to rewrite the equations in term of the running coupling
:math:`a_s` assuming the factorization scale :math:`\mu_F^2` (the inherit scale
of the |PDF|) and the renormalization scale :math:`\mu_R^2` (the inherit scale
for the strong coupling) to be equal. This constraint can however be lifted by a
suitable redefinition of the splitting kernels :cite:`Vogt:2004ns`:

.. math ::
    \gamma^{(1)}(N) &\to \gamma^{(1)}(N) - \beta_0 \ln(\mu_F^2/\mu_R^2) \gamma^{(0)} \\
    \gamma^{(2)}(N) &\to \gamma^{(2)}(N) - 2 \beta_0 \ln(\mu_F^2/\mu_R^2) \gamma^{(1)} - ( \beta_1 \ln(\mu_F^2/\mu_R^2) - \beta_0^2 \ln^2(\mu_F^2/\mu_R^2) )  \gamma^{(0)}


while keeping the evaluation of the strong coupling always at :math:`\mu_R^2`.
Estimating the theoretical uncertainties imposed on |PDF| determination due to
missing higher order corrections using scale variation in the evolution
corresponds to schemes A and B in :cite:`AbdulKhalek:2019ihb`.


Heavy Quark Masses
------------------

In QCD also the heavy quark masses (:math:`m_{c}, m_{b}, m_{t}`) follow a |RGE|
and their values depend on the energy scale at which the quark is probed.
Masses do not play any role in a single flavour patch, but are important in
|VFNS| when more flavour schemes need to be joined (see :doc:`matching
conditions <Matching>`).

EKO implements two strategies for dealing with the heavy quark masses, managed
by the theory card parameter ``HQ``. The easiest and more common option for
PDFs evolution is ``POLE`` mass, where the physical quark masses are
specified as input.

On contrary selecting the option ``MSBAR`` the user can activate the *mass
running* in the |MSbar| scheme, as described in the following
paragraph.

If the initial condition for the mass is not given at a scale coinciding with
the mass itself (i.e. in the input theory card ``Qmh≠mh``),
EKO needs to compute the scale at which the mass running function intersects
the identity function, in order to properly initiate the
:class:`~eko.threshold.ThresholdAtlas` and set the evolution path.

For each heavy quark :math:`h` we solve for :math:`m_h`:

.. math ::
    m_{\overline{MS},h}(m_h^2) = m_h


where the evolved |MSbar| mass is given by:

.. math ::
    m_{\overline{MS},h}(\mu^2) = m_{h,0} \int_{a_s(\mu_{h,0}^2)}^{a_s(\mu^2)} \frac{\gamma(a_s)}{\beta(a_s)} d a_s

and :math:`m_{h,0}` is the given initial condition at the scale
:math:`\mu_{h,0}`. Here there is a subtle complication since the solution
depends on the value :math:`a_s(\mu_{h,0}^2)` which is unknown and depends again
on the threshold path.
To overcome this issue, EKO initialize a temporary instance of the class
:class:`~eko.strong_coupling.StrongCoupling` with a fixed flavor number scheme,
with :math:`n_{f_{ref}}` active flavors at the scale :math:`\mu_{ref}`.

Then we check that, heavy quarks involving a number of active flavors
greater than :math:`n_{f_{ref}}` are given with initial conditions:

.. math ::
    m_h (\mu_h) \ge \mu_h

while the ones related to fewer active flavors follow:

.. math ::
    m_h (\mu_h) \le \mu_h

So for the former initial condition we will find the intercept between |RGE| and the identity
in the forward direction (:math:`m_{\overline{MS},h} \ge \mu_h`) and viceversa for the latter.

In doing so EKO takes advantage of the monotony of the |RGE| solution
:math:`m_{\overline{MS},h}(\mu^2)` with a vanishing limit for :math:`\mu^2
\rightarrow \infty`.

Now, being able to evaluate :math:`a_s(\mu_{h,0}^2)`, there are two ways of
solving the previous integral and finally compute the evolved
:math:`m_{\overline{MS},h}`. In fact, the function :math:`\gamma(a_s)` is the
anomalous QCD mass dimension and, as the :math:`\beta` function, it can be evaluated
perturbatively in :math:`a_s` up to :math:`\mathcal{O}(a_s^3)`:

.. math ::
    \gamma(a_s) &= - \sum\limits_{n=0} \gamma_n a_s^{n+1} \\

Even here it is useful to define :math:`c_k = \gamma_k/\beta_0, k>0`.

Therefore the two solution strategies are:

- ``method = "exact"``: the integral is solved exactly using the expression of
  :math:`\beta,\gamma` up to the specified perturbative order
- ``method = "expanded"``: the integral is approximate by the following expansion:

.. math ::
    m_{\overline{MS},h}(\mu^2) & = m_{h,0} \left ( \frac{a_s(\mu^2)}{a_s(\mu_{h,0}^2)} \right )^{c_0} \frac{j_{exp}(a_s(\mu^2))}{j_{exp}(a_s(\mu_{h,0}^2))} \\
    j_{exp}(a_s) &= 1 + a_s \left [ c_1 - b_1 c_0 \right ] + \frac{a_s^2}{2} \left [c_2 - c_1 b_1 - b_2 c_0 + b_1^2 c_0 + (c_1 - b_1 c_0)^2 \right]


The procedure is iterated on all the heavy quarks, updating the temporary instance
of :class:`~eko.strong_coupling.StrongCoupling` with the computed masses.

To find coeherent solutions and perform the mass running in the correct pathces it
is necessary to always start computing the mass scales closer to :math:`\mu_{ref}`.

Eventually, to ensure that the threshold values are properly set, we add a
consistency check, asserting:

.. math ::
    m_{\overline{MS},h} (m_h) \leq m_{\overline{MS},h+1} (m_h)

We provide the following as an illustrative example of how this procedure works:
when the strong coupling is given with boundary condition :math:`\alpha_s(\mu_{ref}=91, n_{f_{ref}}=5)`
then the heavy quarks initial conditions must satisfy:

.. math ::
    & \mu_{b} \le \mu_{ref} \le \mu_t \\
    & m_c (\mu_c) \le \mu_c \\
    & m_b (\mu_b) \le \mu_b \\
    & m_t (\mu_t) \ge \mu_t

and EKO will start solving the equation :math:`m_{\overline{MS},h}(m_h^2) = m_h`
in the order :math:`h={t,b,c}`.

Since the charm mass will be computed only when both the top and bottom threshold scales
are known, the boundary condition :math:`m_c(\mu_{c})` can be evolved safely below
the scale :math:`m_{\overline{MS},b}` where the solution of
:math:`m_{\overline{MS},c}(m_c^2) = m_c` is sitting.
