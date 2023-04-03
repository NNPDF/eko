Time-Like Evolution
===================

Due to confinement in |QCD| we can not observe partons, such as quarks and gluons,
directly in particle collider experiments.
Instead, stable hadrons are detected which originate from parton interactions.

The fragmentation functions (|FF|) encode the information
on the probability for a hadron carrying a specified momentum fraction to 'fragment'
from a given parton. These functions are non-perturbative and usually require a global |QCD|
analysis of experimental data involving different processes for their reliable
determination. This makes the |FF| similar to |PDF| as both rely
on similar factorization theorems and, thus, on similar |RGE|.
In practice, the relevant Feynman diagrams can indeed be related by a crossing
symmetry which in turn means certain Mandelstam variables become for |FF|
time-like instead of space-like.
The relevant setting in the operator card is thus called ``time_like = True``.

We implement the time-like |DGLAP| anomalous dimensions up to |NNLO| in :class:`~ekore.anomalous_dimensions.unpolarized.time_like`.
The implementation for the |LO| and |NLO| splitting functions is based on :cite:`Mitov:2006wy, Gluck:1992zx` and the implementation for
the |NNLO| splitting functions is based on :cite:`Mitov:2006ic, Moch:2007tx, Almasy:2011eq`.
Supplying new anomalous dimensions and new matching conditions is the only change required for the eko program (e.g. the
solution strategies are unaffected).
