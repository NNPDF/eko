Flavor Space
============

An |EKO| is a rank-4 operator acting both in Flavor Space :math:`\mathcal F`
and momentum fraction space :math:`\mathcal X`.
By Flavor Space :math:`\mathcal F` we mean the 14-dimensional function space that contains
the different |PDF| flavor. Note, that there is an ambiguity concerning the
word "Flavor Basis" which is sometimes referred to as an *abstract* basis
in the Flavor Space, but often the specific basis described here below is meant.

Flavor Basis
------------

Here we use the raw quark flavors along with the gluon and the photon as they correspond to the
operator in the Lagrange density:

.. math ::
    \mathcal F = \mathcal F_{fl} = \span(\gamma, g, u, \bar u, d, \bar d, s, \bar s, c, \bar c, b, \bar b, t, \bar t)

- we deliver the :class:`~eko.output.Output` in this basis, although the flavors are
  slightly differently arranged (Implementation: :data:`here <eko.basis_rotation.flavor_basis_pids>`).
- most cross section programs as well as `LHAPDF <https://lhapdf.hepforge.org/>`_ :cite:`Buckley:2014ana` use this basis
- we will consider this basis as the canonical basis

+/- Basis
---------

Instead of using the raw flavors, we recombine the quark flavors into

.. math ::
    q^\pm = q \pm \bar q

as this is closer to the actual physics: :math:`q^-` corresponds to the valence quark distribution
that e.g. in the proton will carry most of the momentum at large x and :math:`q^+` effectively is the
sea quark distribution:

.. math ::
    \mathcal F \sim \mathcal F_{\pm} = \span(\gamma, g, u^+, u^-, d^+, d^-, s^+, s^-, c^+, c^-, b^+, b^-, t^+, t^-)

- this basis is *not* normalized with respect to the canonical Flavor Basis
- the basis transformation to the Flavor Basis is implemented in
  :meth:`~eko.evolution_operator.flavors.rotate_pm_to_flavor`

QCD Evolution Basis
-------------------

As the gluon is flavor-blind it is handy to solve |DGLAP| not in the flavor basis,
but in the QCD Evolution Basis where instead we need to solve a minimal coupled system.
This is the basis in which DGLAP equations are solved considering only QCD corrections.
The new basis elements can be separated into two major classes: the singlet sector, consisting of the
singlet distribution :math:`\Sigma` and the gluon distribution :math:`g`, and the non-singlet
sector. The non-singlet sector can be again subdivided into three groups: first the full
valence distribution :math:`V`, second the valence-like distributions
:math:`V_3 \ldots V_{35}`, and third the singlet like distributions :math:`T_3 \ldots T_{35}`.
The mapping between the Evolution Basis and the +/- Basis is given by

.. math ::
    \Sigma &= \sum\limits_{j}^6 q_j^+\\
    V &= \sum\limits_{j}^6 q_j^-\\
    V_3 &= u^- - d^-\\
    V_8 &= u^- + d^- - 2 s^-\\
    V_{15} &= u^- + d^- + s^- - 3 c^-\\
    V_{24} &= u^- + d^- + s^- + c^- - 4 b^-\\
    V_{35} &= u^- + d^- + s^- + c^- + b^- - 5 t^-\\
    T_3 &= u^+ - d^+\\
    T_8 &= u^+ + d^+ - 2 s^+\\
    T_{15} &= u^+ + d^+ + s^+ - 3 c^+\\
    T_{24} &= u^+ + d^+ + s^+ + c^+ - 4 b^+\\
    T_{35} &= u^+ + d^+ + s^+ + c^+ + b^+ - 5 t^+\\
    \mathcal F \sim \mathcal F_{ev} &= \span(\gamma, g, \Sigma, V, V_{3}, V_{8}, V_{15}, V_{24}, V_{35}, T_{3}, T_{8}, T_{15}, T_{24}, T_{35})


- the associated numbers to the valence-like and singlet-like non-singlet distributions
  :math:`k` follow the common group-theoretical notation :math:`k = n_f^2 - 1`
  where :math:`n_f` denotes the incorporated number of quark flavors
- this basis is *not* normalized with respect to the canonical Flavor Basis
- the basis transformation from the Flavor Basis is implemented in
  :data:`~eko.basis_rotation.rotate_flavor_to_evolution`
- the photon is just a spectator and does not couple to anyone

Intrinsic QCD Evolution Bases
-----------------------------

However, the QCD Evolution Basis is not yet the most decoupled basis if we consider intrinsic evolution.
The intrinsic distributions do *not* participate in the |DGLAP| equation but instead evolve with a unity operator:
this makes, e.g. :math:`T_{15}` a composite object in a evolution range below the charm mass.
Instead, we will keep the non participating distributions here in their :math:`q^\pm` representation.
The Intrinsic QCD Evolution Bases will explicitly depend on the number of light flavors :math:`n_f`.
For :math:`n_f=3` we define (the other cases are defined analogously):

.. math ::
    \mathcal F \sim  \mathcal F_{iev,3} = \span(\gamma, g, \Sigma_{(3)}, V_{(3)}, V_3, V_8, T_3, T_8, c^+, c^-, b^+, b^-, t^+, t^-)

where we defined :math:`\Sigma_{(3)} = \sum\limits_{j=1}^3 q_j^+` and :math:`V_{(3)} = \sum\limits_{j=1}^3 q_j^-`
(not to be confused with the usual :math:`V_3`).

- for :math:`n_f=6` the Intrinsic QCD Evolution Basis coincides with the QCD Evolution Basis: :math:`\mathcal F_{iev,6} = \mathcal F_{ev}`
- this basis is *not* normalized with respect to the canonical Flavor Basis
- the basis transformation from the Flavor Basis is implemented in
  :meth:`~eko.evolution_operator.flavors.pids_from_intrinsic_evol`
- note that for the case of non-intrinsic component the higher elements in :math:`\mathcal F_{ev}` do become linear dependent
  to other basis vectors (e.g. :math:`\left. T_{15}\right|_{c^+ = 0} = \Sigma`) but are non zero - instead in :math:`\mathcal F_{iev,3}`
  this direction vanishes
- the photon is just a spectator and does not couple to anyone


QED :math:`\otimes` QCD Evolution Basis
---------------------------------------

In presence of QED corrections to DGLAP evolution equations, 
the QCD Evolution basis does not decouple the distributions 
as it was for the pure QCD evolution.
In this case the QED :math:`\otimes` QCD evolution basis that performs the maximal decoupling is given by:

.. math ::
  \Sigma_u & = u^+ + c^+ + t^+ \\
  \Sigma_d & = d^+ + s^+ + b^+ \\
  \Sigma &= \Sigma_u + \Sigma_d \\
  \Delta_\Sigma &= \Sigma_u - \Sigma_d \\
  V_u & = u^- + c^- + t^- \\
  V_d & = d^- + s^- + b^- \\ 
  V &= V_u + V_d \\
  \Delta_V &= V_u - V_d \\
  T_1^u &=u^+ - c^+ \\
  T_2^u &=u^+ + c^+ - 2t^+ \\
  T_1^d &=d^+ - s^+ \\
  T_2^d &=d^+ + s^+ - 2b^+ \\
  V_1^u &=u^- - c^- \\
  V_2^u &=u^- + c^- - 2t^- \\
  V_1^d &=d^- - s^- \\
  V_2^d &=d^- + s^- - 2b^- \\
  \mathcal F \sim  \mathcal F_{QED} &= \span(\gamma, g, \Sigma, \Delta_\Sigma, V, \Delta_V, T_1^u, T_2^u, T_1^d, T_2^d, V_1^u, V_2^u, V_1^d, V_2^d)


- :math:`T_1` and :math:`T_2` are just arbitrary names and do not follow any group representation like it was for the :ref:`theory/FlavorSpace:QCD Evolution Basis`
- this basis is *not* normalized with respect to the canonical Flavor Basis
- The singlet :math:`\Sigma` is just the QCD singlet 
- The valence :math:`V` is just the QCD valence 

 




Other Bases
-----------

In an |PDF| fitting environment sometimes yet different bases are used to enforce or improve positivity
of the |PDF| :cite:`Candido:2020yat`. E.g. :cite:`Giele:2002hx` uses

.. math ::
    u_v = u^-, d_v = d^-, L_+ = 2(\bar u + \bar d), L_- = \bar d - \bar u, s^+, c^+, b^+, g

Operator Bases
--------------

An |EKO| :math:`\mathbf E` is an operator in the Flavor Space :math:`\mathcal F` mapping one vector onto an other:

.. math ::
    \mathbf E \in \mathcal F \otimes \mathcal F

since evolution can (and will) mix flavors. To specify the basis for these operators we need to specify the basis
for both the input and output space.

Operator Flavor Basis
^^^^^^^^^^^^^^^^^^^^^

- here we mean :ref:`theory/FlavorSpace:Flavor Basis` both in the input and the output space
- the :class:`~eko.output.Output` is delivered in this basis
- this basis has :math:`(2n_f+ 1)^2 = 13^2 = 169` elements
- this basis can span arbitrary thresholds

Operator Anomalous Dimension Basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- here we mean the true underlying physical basis where elements correspond to the different splitting functions,
  i.e. :math:`\mathbf{E}_S, E_{ns,v}, E_{ns,+}, E_{ns,-}`
- this basis has 4 elements in |LO|, 6 elements in |NLO| and its maximum 7 elements after |NNLO|
- this basis can *not* span any threshold but can only be used for a *fixed* number of flavors
- all actual computations are done in this basis

Operator Intrinsic Evolution Basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- here we mean :ref:`theory/FlavorSpace:Intrinsic Evolution Bases` both in the input and the output space
- this basis does **not** coincide with the :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis` as the decision on which operator of that
  basis is used is a non-trivial decision - see :doc:`Matching`
- this basis has :math:`2n_f+ 3 = 15` elements
- this basis can span arbitrary thresholds
