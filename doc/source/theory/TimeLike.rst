Time Like Evolution
===================

In particle collider experiments with high enough energy involving hadrons, the 
hard scattering interactions involve quarks and gluons, which can be studied through 
Feynman diagrams, however these quarks and gluons are not the particles that are 
detected in the final state due to confinement. Instead, hadrons are detected 
which originate from the quarks and gluons involved in the interaction.

The fragmentation functions are precisely the functions which encode information 
on the probability for a hadron carrying a specified momentum fraction to 'fragment' 
from a given quark/gluon. These functions are non-perturbative and require global QCD 
analysis of experimental data involving lots of different processes for their effectivel
determination.

The above explanation clearly show similarity to Parton Distribution Functions, in that, 
both categories of functions are probability density functions. A common feature amongst them
is a reliance on the resolution scale :math:`Q^2`, and therefore this brings about the 
need to be able to evolve the operators in question to desired level of the resolution scale.

The difference between Parton Distribution Functions and Fragmentation Functions is in the |DGLAP|
equations that are used. Fragmentation Functions require the time-like DGLAP equations and |EKO|
is able to compute them by using the ``time_like = True`` in the operator card.
We implement the time-like |DGLAP| splitting functions upto |NNLO| in the  :class:`ekore.anomalous_dimensions.unpolarized.time_like`.
The implementation for the |LO| and |NLO| splitting functions is based on :cite:`Mitov:2006wy, Gluck:1992zx` and the implementation for
the |NNLO| splitting functions is based on :cite:`Mitov:2006ic, Moch:2007tx, Almasy:2011eq`.