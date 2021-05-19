Flavor Space
============

An |EKO| is a rank-4 operator acting both in Flavor Space :math:`\mathcal F`
and momentum fraction space :math:`\mathcal X`.
By Flavor Space :math:`\mathcal F` we mean the 13-dimensional function space that contains
the different |PDF| flavor. Note, that there is an ambiguity concerning the
word "Flavor Basis" which is sometimes refered to as an *abstract* basis
in the Flavor Space, but often the specific basis described here below is meant.

Flavor Basis
------------

Here we use the raw quark flavors along with the gluon as they correspond to the
operator in the Lagrange density:

.. math ::
    \mathcal F = \text{span}(g, u, \bar u, d, \bar d, s, \bar s, c, \bar c, b, \bar b, t, \bar t)

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
    \mathcal F \sim \text{span}(g, u^+, u^-, d^+, d^-, s^+, s^-, c^+, c^-, b^+, b^-, t^+, t^-)

- this basis is *not* normalized with respect to the canonical Flavor Basis
- the basis transformation to the Flavor Basis is implemented in
  :meth:`~eko.evolution_operator.flavor.rotate_pm_to_flavor`

Evolution Basis
---------------

As the gluon is flavor-blind it is handy to solve |DGLAP| not in the flavor basis,
but in the Evolution Basis where instead we need to solve a minimal coupled system.
The new basis elements can be seperated into two major classes: the singlet sector, consisting of the
singlet distribution :math:`\Sigma` and the gluon distribution :math:`g`, and the non-singlet
sector. The non-singlet sector can be again subdivided into three groups: first the full
valence distribution :math:`V`, second the valence-like distributions
:math:`V_3 \ldots V_{35}`, and third the singlet like distributions :math:`T_3 \ldots T_{35}`.
The mapping between the Evolution Basis and the +/- Basis is given by

.. math ::
    \Sigma &= \sum\limits_{j} q_j^+\\
    V &= \sum\limits_{j} q_j^-\\
    V_3 &= u^- - d^-\\
    V_8 &= u^- + d^- - 2 s^-\\
    V_{15} &= u^- + d^- + s^- - 3 c^-\\
    V_{24} &= u^- + d^- + s^- + c^- - 4 b^-\\
    V_{35} &= u^- + d^- + s^- + c^- + b^- - 5 t^-\\
    T_3 &= u^+ - d^+\\
    T_8 &= u^+ + d^+ - 2 s^+\\
    T_{15} &= u^+ + d^+ + s^+ - 3 c^+\\
    T_{24} &= u^+ + d^+ + s^+ + c^+ - 4 b^+\\
    T_{35} &= u^+ + d^+ + s^+ + c^+ + b^+ - 5 t^+


- the associated numbers to the valence-like and singlet-like non-singlet distributions
  :math:`k` follow the common group-theoretical notation :math:`k = n_f^2 - 1`
  where :math:`n_f` denotes the incorporated number of quark flavors
- this basis is *not* normalized with respect to the canonical Flavor Basis, this means that in the
  final step before the :class:`~eko.output.Output` is created, the elements have to be normalized
- the basis transformation from the Flavor Basis is implemented in
  :data:`~eko.basis_rotation.rotate_flavor_to_evolution`

Other Bases
-----------

In an |PDF| fitting environment sometimes yet different bases are used to enforce or improve positivity
of the |PDF| :cite:`Candido:2020yat`. E.g. :cite:`Giele:2002hx` uses

.. math ::
    u_v = u^-, d_v = d^-, L_+ = 2(\bar u + \bar d), L_- = \bar d - \bar u, s^+, c^+, b^+, g

Operator Bases
--------------

An |EKO| :math:`\mathbf E` is operator in the Flavor Space :math:`\mathcal F` mapping one vector onto an other:

.. math ::
    \mathbf E \in \mathcal F \otimes \mathcal F

so to specify the basis for these operators we need to specify the basis for both the input and output space:

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
- this basis has 4 elements in |LO|, 6 elements in |NLO| and its maximum 7 elemets after |NNLO|
- this basis can *not* span any threshold but can only be used for a *fixed* number of flavors
- all actual computations are done in this basis

Operator Evolution Basis
^^^^^^^^^^^^^^^^^^^^^^^^

- here we mean :ref:`theory/FlavorSpace:Evolution Basis` both in the input and the output space
- this basis does **not** coincide with the :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis` as the decision on which operator of that
  basis is used is a non-trivial decision - see :doc:`Matching`
- this basis has :math:`2n_f+ 3 = 15` elements
- this basis can span arbitrary thresholds