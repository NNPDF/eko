Flavor Space
============

An |EKO| is rank-4 operator both in flavor and momentum fraction space.
By Flavor Space we mean the 13-dimensional function space that contains
the different |PDF| flavor. Note that there is an abiguity concerning the
word "Flavor Basis" which is sometimes refered to as an *abstract* basis
in the Flavor Space, but often the specific basis described here below is meant.

Flavor Basis
------------

Here we use the raw quark flavors along with the gluon as they correspond to the
operator in the Lagrange density:

.. math ::
    g, u, \bar u, d, \bar d, s, \bar s, c, \bar c, b, \bar b, t, \bar t

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
    g, u^+, u^-, d^+, d^-, s^+, s^-, c^+, c^-, b^+, b^-, t^+, t^-

- this basis is *not* normalized with respect to the canonical flavor basis
- the basis transformation to the Flavor Basis is implemented in
  :meth:`~eko.operator.flavor.rotate_pm_to_flavor`

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
- this basis is *not* normalized with respect to the canonical flavor basis, this means that in the
  final step before the :class:`~eko.output.Output` is created, the elements have to be normalized
- the basis transformation from the Flavor Basis is implemented in
  :data:`~eko.basis_rotation.rotate_flavor_to_evolution`

Other Bases
-----------

In an |PDF| fitting environment sometimes yet different bases are used to enforce or improve positivity
of the |PDF| :cite:`Candido:2020yat`. E.g. :cite:`Giele:2002hx` uses

.. math ::
    u_v = u^-, d_v = d^-, L_+ = 2(\bar u + \bar d), L_- = \bar d - \bar u, s^+, c^+, b^+, g