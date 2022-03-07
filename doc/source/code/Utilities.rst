Utility Classes
===============

Apart from the :doc:`operator <Operators>` classes, `eko` also provides some utility classes which are e.g. used in |yadism|

- :class:`eko.thresholds.ThresholdsAtlas`

  -  Implementation of the flavor number scheme and the quark thresholds both for
     the :class:`eko.strong_coupling.StrongCoupling` and the :doc:`operators <../theory/Matching>`

  When running in |VFNS| it is important to specify the number of flavors active at each given scale, since evolution path
  can be different depending of the chosen setting. This path is determined by :meth:`eko.thresholds.ThresholdsAtlas.path`.

  Let us consider an example to better illustrate how it works.
  Imagine to have a boundary condition ``q2_ref=64, nf_ref=3`` and would like to evolve your object (|PDF| or :math:`a_s`) to a lower
  scale (say ``Q=7``) but giving different number of active flavors to the final state and having set the heavy quarks mass thresholds
  at: ``mc=2, mb=3, mt=4``. In this case some possible paths are:

    - a path to ``Q=7`` with ``nf=6`` (default) which is given by the following list of :class:`eko.thresholds.PathSegment` :

      .. code-block::

        [PathSegment(64 -> 4, nf=3), PathSegment(4 -> 9, nf=4), PathSegment(9 -> 16, nf=5), PathSegment(16 -> 49, nf=6)]

      and it's determined according to the list of heavy quark thresholds.

    - A path to ``Q=7`` enforcing ``nf=3`` which will simply contain a single :class:`eko.thresholds.PathSegment`:

      .. code-block::

        [PathSegment(64 -> 49, nf=3)]

    - A path to ``Q=7`` enforcing ``nf=4``:

      .. code-block::

        [PathSegment(64 -> 4, nf=3), PathSegment(4 -> 49, nf=4)]

  As first step, you always go back to the closest matching scale (closest in ``nf``),
  running in ``nf=3``, since that is what has been specified in the boundary conditions.
  Then you are back in the normal flow, and you cross all the other thresholds going according to the prescription given
  by the :class:``eko.thresholds.ThresholdAtlas``.
  You can noticed that :meth:`eko.thresholds.ThresholdsAtlas.path` are always ordered according to `nf` and not `Q2` scales.
  This property is used in |VFNS| to determine if a path is backward or not modifying the way the different patches are matched.

.. include:: IO-tabs/ThresholdConfig.rst


- :class:`eko.strong_coupling.StrongCoupling` Implementation of the :ref:`theory/pQCD:strong coupling`

- :class:`eko.interpolation.InterpolatorDispatcher` Implementation of the :doc:`../theory/Interpolation`
