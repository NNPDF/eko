pQCD ingredients
================

Strong Coupling
---------------

Implementation: :class:`~eko.couplings.Couplings`.

We use perturbative |QCD| with the running coupling
:math:`a_s(\mu_R^2) = \alpha_s(\mu_R^2)/(4\pi)` given at 5-loop by
:cite:`Herzog:2017ohr,Luthe:2016ima,Baikov:2016tgj,Chetyrkin:2017bjc,Luthe:2017ttg`

.. math ::
    \frac{da_s(\mu_R^2)}{d\ln\mu_R^2} = \beta(a_s(\mu_R^2)) \
    = - \sum\limits_{n=0} \beta_n a_s^{n+2}(\mu_R^2)

It is useful to define in addition :math:`b_k = \beta_k/\beta_0, k>0`.

We implement two different strategies to solve the |RGE|:

- ``method="exact"``: Solve using :func:`scipy.integrate.solve_ivp`.
  In |LO| we fall back to the expanded solution as this is already the true solution.
- ``method="expanded"``: using approximate solutions:

.. math ::
    a^{\text{LO}}_s(\mu_R^2) &= \frac{a_s(\mu_0^2)}{1 + a_s(\mu_0^2) \beta_0 L_{\mu}} \\
    a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2) &= a^{\text{LO}}_s(\mu_R^2)-b_1 \left[a^{\text{LO}}_s(\mu_R^2)\right]^2 \ln\left(1+a_s(\mu_0^2) \beta_0 L_{\mu}\right) \\
    a^{\text{NNLO}}_{s,\text{exp}}(\mu_R^2) &= a^{\text{LO}}_s(\mu_R^2)\left[1 + a^{\text{LO}}_s(\mu_R^2)\left(a^{\text{LO}}_s(\mu_R^2) - a_s(\mu_0^2)\right)(b_2 - b_1^2) \right.\\
                                        & \hspace{60pt} \left. + a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2) b_1 \ln\left(a^{\text{NLO}}_{s,\text{exp}}(\mu_R^2)/a_s(\mu_0^2)\right)\right] \\
    a^{\text{N3LO}}_{s,\text{exp}}(\mu_R^2) &= a^{\text{NNLO}}_s(\mu_R^2) + \frac{a^{\text{LO}}_s(\mu_R^2)^4}{2 b_0^3} \left\{ \right. \\
                & -2 b_1^3 L_{0}^3 + 5 b_1^3 L_{\text{LO}}^2 + 2 b_1^3  L_{\text{LO}}^3 + b_1^3 L_{0}^2 \left(5 + 6  L_{\text{LO}} \right) \\
                & + 2 b_0 b_1  L_{\text{LO}} \left[ b_2 + 2 \left(b_1^2 - b_0 b_2 \right) L_{\mu} a_s(\mu_0^2) \right] \\
                & - b_0^2 L_{\mu} a_s(\mu_0^2) \left[ -2 b_1 b_2 + 2 b_0 b_3 + \left( b_1^3 - 2 b_0 b_1 b_2 + b_0^2 b_3 \right) L_{\mu} a_s(\mu_0^2) \right] \\
                & - 2 b_1 L_{0} \left[ 5 b_1^2  L_{\text{LO}} + 3 b_1^2  L_{\text{LO}}^2 + b_0 \left[b_2 + 2 \left(b_1^2 - b_0 b_2\right) L_{\mu} a_s(\mu_0^2)\right] \right ] \\
                & \left. \right\}

being:

.. math ::
    L_{\mu} &= \ln(\mu_R^2/\mu_0^2) \\
    L_{0} &= \ln(a_s(\mu_0^2)) \\
    L_{\text{LO}} &= \ln(a^{\text{LO}}_s(\mu_R^2)) \\

When the renormalization scale crosses a flavor threshold matching conditions
have to be applied :cite:`Schroder:2005hy,Chetyrkin:2005ia`.
In particular, the matching involved in the change from :math:`n_f` to :math:`n_f-1` schemes
is presented in equation 3.1 of :cite:`Schroder:2005hy` for |MSbar| masses, while the
same expression for POLE masses is reported in Appendix A.
For this reason the boundary conditions of :class:`eko.couplings.Couplings`
can be specified at ``scale_ref`` along with ``nf_ref`` and, the computed result can
depend on the number of flavors at the target scale, see :meth:`eko.couplings.Couplings.a_s`
An example how the evolution path is determined is given :doc:`here</code/Utilities/>`.


QCD Splitting Functions
-----------------------

In the case in which only the |QCD| corrections are considered, the Altarelli-Parisi splitting kernels can be expanded in powers of the strong
coupling :math:`a_s(\mu^2)` and are given by :cite:`Moch:2004pa,Vogt:2004mw`

.. math ::
    \mathbf{P}(x,a_s(\mu^2)) &= \sum\limits_{j=0} a_s^{j+1}(\mu^2) \mathbf{P}^{(j)}(x) \\
    {\gamma}^{(j)}(N) &= -\mathcal{M}[\mathbf{P}^{(j)}(x)](N)

Note the additional minus in the definition of :math:`\gamma`.

Polarized Splitting Functions
-----------------------------

Polarized Altarelli-Parisi splitting kernels are implemented up to |NNLO| and expanded in powers of the strong coupling as in the previous section.
They are used to evolve longitudinally polarized parton distribution functions.
Unlike in the unpolarized case, where the probability of the splitting describes the momentum of parent and daughter partons with averaged spins,
the polarized splitting functions describe the parent and daughter momentums along with their spins
and thus take into account positive or negative helicities.
Throughout, the anomalous dimensions are defined as above and are represented with :math:`\gamma` and not :math:`\Delta\gamma` just like in the unpolarized case.

The |LO| and |NLO| kernels are given in :cite:`Gluck:1995yr` and the |NNLO| in :cite:`Moch:2014sna` and :cite:`Moch:2015usa`.

At |LO|, the non-singlet is the same in both the polarized and unpolarized case.
Due to helicity conservation, the first moment of the anomalous dimension is :math:`\gamma^{(0)}_{qq} (N=1) = \gamma^{(0)}_{qg} (N=1) = 0`.

At |NLO|, the singlet entry of the quark-quark anomalous dimension can be decomposed into the pure singlet
(consisting of the flavour independent quark-quark and quark-antiquark anomalous dimensions) and the plus flavour asymmetry non-singlet:

.. math ::
    \gamma^{(1)}_{qq} =\gamma^{(1)}_{ps} + \gamma^{(1)}_{ns,+}

The non-singlet sector in the polarized case swaps the plus and minus non-singlet relative to the unpolarized case.
This is because the polarized non-singlet splitting functions are defined as the difference between the probability of the polarized parton splitting 
into daughter partons of the same flavour and same helicity and daughters of a different flavours and opposite helicity. 
The first moments of the anomalous dimensions are:

.. math ::
    \gamma^{(1)}_{ns,+} (N=1) &= 0 \\
    \gamma^{(1)}_{qq} (N=1) &= 24 C_F T_R \\
    \gamma^{(1)}_{qg} (N=1) &= 0  \\

At |NNLO| the non-singlet is further decomposed into the helicity difference quark-antiquark anomalous dimension called the valence polarized non-singlet and defined as:

.. math ::
    \gamma^{(2)}_{ns,v} =\gamma^{(2)}_{ns,-} + \gamma^{(2)}_{ns,s}

where :math:`\gamma^{(2)}_{ns,-}` is the minus flavour asymmetry non-singlet and :math:`\gamma^{(2)}_{ns,s}` the sea-like polarized non-singlet.
The singlet entry :math:`\gamma^{(2)}_{qq}` is defined as above in the |NLO| case.

Finally the violation of the axial current conservation :math:`\bar{\psi} \gamma_\mu \gamma_5 \bar{\psi}` only through 
loop corrections impose the following relations to the singlet splittings at all orders :cite:`Moch:2014sna` :

.. math ::
     \gamma^{(n)}_{gg} & = - \beta_n \\
     \gamma^{(n)}_{ps} & = - 2 n_f \gamma^{(n-1)}_{gq}

Unified Splitting Functions
---------------------------

When the |QED| corrections are taken into account, |DGLAP| equation take the form

.. math ::
    \mathbf{P}=\mathbf{\tilde{P}}+\mathbf{\bar{P}}

where :math:`\mathbf{\tilde{P}}` are the usual |QCD| splitting kernels defined in the previous section,
while :math:`\mathbf{\bar{P}}` are given by

.. math ::
    \mathbf{\bar{P}} = a \mathbf{P}^{(0,1)} + a_s a \mathbf{P}^{(1,1)} +
   a^2 \mathbf{P}^{(0,2)} + \dots

where :math:`a = \alpha/(4\pi)`.
The expression of the pure |QED| and of the mixed |QED| :math:`\otimes` |QCD| splitting kernels are given in
:cite:`deFlorian:2015ujt,deFlorian:2016gvk`


Order specification
-------------------

In the code ``order=tuple(int,int)`` specifies the |QCD| and |QED| perturbative orders of the splitting functions in terms
of :math:`a_s = \alpha_s/(4\pi)` and :math:`a_{em} = \alpha_{em}/(4\pi)`. The available perturbative expansions are the following:

- ``order=(n,0)``: with :math:`n=1,2,3,4` correspond to the pure |QCD| evolution at |LO|, |NLO|, |NNLO| and |N3LO| in which the |QCD| splitting functions are expanded up to :math:`\mathcal{O}(a_s^n)` and the strong coupling is evolved using the n-th coefficient of the beta function, i.e. :math:`\beta_{n-1}`.
- ``order=(n,m)``; with :math:`n=1,2,3,4` and :math:`m=1,2` corresponds to the mixed |QED| :math:`\otimes` |QCD| evolution in which the splitting functions are expanded up to :math:`\mathcal{O}(a_s^na_{em}^m)`, the stromg coupling is evolved using up to the n-th coefficient of the beta function and the electromagnetic coupling is kept fixed.

Observe that the case :math:`n=0` is not allowed, since it would correspond to the pure |QED| evolution or (if :math:`m > 0`) no evolution at all.

Sum Rules
---------

The Altarelli-Parisi Splitting functions have to satisfy certain sum rules. In fact |QED| :math:`\otimes` |QCD|
interactions preserve fermion number, therefore

.. math ::
    \int_0^1dx P_{ns,q}^-(x)=0

Moreover, the conservation of the proton's momentum implies that

.. math ::
    \int_0^1dx x (2n_dP_{dg}(x)+2n_uP_{ug}(x)+P_{\gamma g}(x)+P_{gg}(x))=0

.. math ::
    \int_0^1dx x (2n_dP_{d\gamma}(x)+2n_uP_{u\gamma}(x)+P_{\gamma \gamma}(x)+P_{g\gamma}(x))=0

.. math ::
    \int_0^1dx x \Bigl(\sum_{q_i=q,\bar{q}} P_{q_iq_j}(x)+P_{\gamma q_j}(x)+P_{gq_j}(x)\Bigr)=0

The reason why multiple conservation equations follow from a single conserved
quantity (i.e. proton's momentum) is that one is free to choose a border
condition in which there is only one parton, e.g. the gluon, and the momentum
should be preserved.
This is just a simple way to consider that anomalous dimensions are actually
operators, and the conservation thus apply element by element in the first
dimension (summing over the second one only).

Using the definition of anomalous dimensions the sum rules are written as:

.. math ::
    \gamma_{ns}^-(N=1)=0

.. math ::
    \bigl(2n_d\gamma_{dg}+2n_u\gamma_{ug}+\gamma_{\gamma g}+\gamma_{gg}\bigr)(N=2)=0

.. math ::
    \bigl(2n_d \gamma_{d\gamma}+2n_u \gamma_{u\gamma}+ \gamma_{\gamma \gamma}+ \gamma_{g\gamma})(N=2)=0

.. math ::
    \Bigl(\gamma_{ns,q}^+ +2n_u\gamma^S_{uq}+2n_d\gamma^S_{dq} + \gamma_{\gamma q}+\gamma_{gq}\Bigr)(N=2)=0

that must be satisfied order by order in perturbation theory.


Heavy Quark Masses
------------------

In |QCD| also the heavy quark masses (:math:`m_{c}, m_{b}, m_{t}`) follow a |RGE|
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
    m_{\overline{MS},h}(\mu^2) = m_{h,0} \exp \left[ - \int_{a_s(\mu_{h,0}^2)}^{a_s(\mu^2)} \frac{\gamma_m(a_s)}{\beta(a_s)} d a_s \right ]

and :math:`m_{h,0}` is the given initial condition at the scale
:math:`\mu_{h,0}`. Here there is a subtle complication since the solution
depends on the value :math:`a_s(\mu_{h,0}^2)` which is unknown and depends again
on the threshold path.
To overcome this issue, EKO initialize a temporary instance of the class
:class:`~eko.couplings.Couplings` with a fixed flavor number scheme,
with :math:`n_{f_{ref}}` active flavors at the scale :math:`\mu_{ref}`.

Then we check that, heavy quarks involving a number of active flavors
greater than :math:`n_{f_{ref}}` are given with initial conditions:

.. math ::
    m_h (\mu_h) \ge \mu_h

while the ones related to fewer active flavors follow:

.. math ::
    m_h (\mu_h) \le \mu_h

So for the former initial condition we will find the intercept between |RGE| and the identity
in the forward direction (:math:`m_{\overline{MS},h} \ge \mu_h`) and vice versa for the latter.

In doing so EKO takes advantage of the monotony of the |RGE| solution
:math:`m_{\overline{MS},h}(\mu^2)` with a vanishing limit for :math:`\mu^2
\rightarrow \infty`.

Now, being able to evaluate :math:`a_s(\mu_{h,0}^2)`, there are two ways of
solving the previous integral and finally compute the evolved
:math:`m_{\overline{MS},h}`. In fact, the function :math:`\gamma_m(a_s)` is the
anomalous |QCD| mass dimension and, as the :math:`\beta` function, it can be evaluated
perturbatively in :math:`a_s` up to :math:`\mathcal{O}(a_s^4)`:

.. math ::
    \gamma_m(a_s) &= \sum\limits_{n=0} \gamma_{m,n} a_s^{n+1} \\

Even here it is useful to define :math:`c_k = \gamma_{m,k}/\beta_0, k \ge 0`.

Therefore the two solution strategies are:

- ``method = "exact"``: the integral is solved exactly using the expression of
  :math:`\beta,\gamma_m` up to the specified perturbative order
- ``method = "expanded"``: the integral is approximate by the following expansion:

.. math ::
    m_{\overline{MS},h}(\mu^2) & = m_{h,0} \left ( \frac{a_s(\mu^2)}{a_s(\mu_{h,0}^2)} \right )^{c_0} \frac{j_{exp}(a_s(\mu^2))}{j_{exp}(a_s(\mu_{h,0}^2))} \\
    j_{exp}(a_s) &= 1 + a_s \left [ c_1 - b_1 c_0 \right ] \\
                 & + \frac{a_s^2}{2} \left [c_2 - c_1 b_1 - b_2 c_0 + b_1^2 c_0 + (c_1 - b_1 c_0)^2 \right] \\
                 & + \frac{a_s^3}{6} [ -2 b_3 c_0 - b_1^3 c_0 (1 + c_0) (2 + c_0) - 2 b_2 c_1 \\
                 & - 3 b_2 c_0 c_1 + b_1^2 (2 + 3 c_0 (2 + c_0)) c_1 + c_1^3 + 3 c_1 c_2 \\
                 & + b_1 (b_2 c_0 (4 + 3 c_0) - 3 (1 + c_0) c_1^2 - (2 + 3 c_0) c_2) + 2 c_3 ]


The procedure is iterated on all the heavy quarks, updating the temporary instance
of :class:`~eko.couplings.Couplings` with the computed masses.

To find coherent solutions and perform the mass running in the correct patches it
is necessary to always start computing the mass scales closer to :math:`\mu_{ref}`.

Eventually, to ensure that the threshold values are properly set, we add a
consistency check, asserting that the :math:`m_{\overline{MS},h}` are properly sorted.

Note that also for |MSbar| mass running when the heavy matching scales are
crossed we need to apply non trivial matching from order
:math:`\mathcal{O}(a_s^2)` as described here :cite:`Liu:2015fxa`.

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

Since the charm mass will be computed only when both the top and bottom matching scales
are known, the boundary condition :math:`m_c(\mu_{c})` can be evolved safely below
the scale :math:`m_{\overline{MS},b}` where the solution of
:math:`m_{\overline{MS},c}(m_c^2) = m_c` is sitting.
