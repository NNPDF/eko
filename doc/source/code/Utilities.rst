Utility Classes
===============

Apart from the :doc:`operator <Operators>` classes, `eko` also provides some utility classes which are e.g. used in |yadism|

- :class:`eko.thresholds.ThresholdsAtlas`

  -  Implementation of the flavor number scheme and the quark thresholds both for
     the :class:`eko.strong_coupling.Couplings` and the :doc:`operators <../theory/Matching>`

  When running in |VFNS| it is important to specify the number of flavors active at each given scale, since the evolution path
  can be different depending of the chosen setting. This path is determined by :meth:`eko.thresholds.ThresholdsAtlas.path`.

  Let us consider two examples to better illustrate how it works.
  Imagine to have a boundary condition ``q2_ref=1``, ``nf_ref=3`` with heavy quarks mass thresholds
  at: ``mc=2``, ``mb=3``, ``mt=4`` and would like to evolve your object (|PDF| or :math:`a_s`) to an higher
  scale (say ``q2_to=49``). The corresponding ``ThresholdsAtlas`` will look like:

  .. code-block::

    ThresholdsAtlas [0.00e+00 - 4.00e+00 - 9.00e+00 - 16.00e+00 - inf], ref=1.000000000000001 @ 3

  and the evolution path will be given by:

  .. code-block::

        [PathSegment(1 -> 4, nf=3), PathSegment(4 -> 9, nf=4), PathSegment(9 -> 16, nf=5), PathSegment(16 -> 49, nf=6)]

  where automatically the number of flavors active in each :class:`eko.thresholds.PathSegment` is determined according with the
  specified thresholds scales.

  However some more complicated situations can arise when the boundary conditions are not given with the same prescription
  of the :class:`eko.thresholds.ThresholdAtlas`. Let's now consider as boundary condition ``q2_ref=64`` with ``nf_ref=3``:

  .. code-block::

    ThresholdsAtlas [0.00e+00 - 4.00e+00 - 9.00e+00 - 16.00e+00 - inf], ref=64.000000000000001 @ 3

  Again we would like to evolve to ``q2_to=49`` but now giving a different number of active flavors to the final scale.
  Some possible paths are:

    - a path to ``q2_to=49`` with ``nf=6`` which is given by the following list of :class:`eko.thresholds.PathSegment` :

      .. code-block::

        [PathSegment(64 -> 4, nf=3), PathSegment(4 -> 9, nf=4), PathSegment(9 -> 16, nf=5), PathSegment(16 -> 49, nf=6)]

      and it's determined according to the list of heavy quark thresholds. This path is default option when the argument
      ``nf_to`` is not given.

    - A path to ``q2_to=49`` enforcing ``nf=4``:

      .. code-block::

        [PathSegment(64 -> 4, nf=3), PathSegment(4 -> 49, nf=4)]

    - A path to ``q2_to=49`` enforcing ``nf=3`` which will simply contain a single :class:`eko.thresholds.PathSegment`:

      .. code-block::

        [PathSegment(64 -> 49, nf=3)]


  In the first two cases as first step, you go back to the closest matching scale (closest in `nf`),
  running in ``nf=3``, since that is what has been specified in the boundary conditions.
  Then you are back in the flavor flow, and you cross all the other thresholds going according to the prescription given
  by the :class:`eko.thresholds.ThresholdAtlas`.
  While in the third example you go directly to the final scale, since there is no matching scale in the middle.
  You can notice that :meth:`eko.thresholds.ThresholdsAtlas.path` are always ordered according to `nf` and not `q2` scales.
  This property is used in |VFNS| to determine if the matchings are done for an increasing or decreasing number of
  light flavors.


- :class:`eko.strong_coupling.StrongCoupling`: Implementation of the :ref:`theory/pQCD:strong coupling`

- :class:`eko.interpolation.InterpolatorDispatcher`: Implementation of the :doc:`../theory/Interpolation`
