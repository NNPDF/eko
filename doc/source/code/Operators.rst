Operator Classes
=================

The classes are nested as follows:

.. graphviz::
    :name: operators
    :caption: nesting of the operator classes: solid lines means "has many",
                dashed lines means "evolves into"
    :align: center

    digraph G {
        bgcolor = transparent

        node [shape=box];
        OpMember [label="OpMember"];
        ndarray [label="np.ndarray"];
        MatchingCondition [label="MatchingCondition" ];
        PhysicalOperator [label="PhysicalOperator"];
        Operator [label="Operator" ];
        OME [label="OME" ];

        Operator -> PhysicalOperator [weight=100,style=dashed];
        PhysicalOperator -> ndarray [style=dashed];
        OME -> MatchingCondition [weight=100,style=dashed];
        MatchingCondition -> ndarray [style=dashed];
        Operator -> OpMember;
        OpMember -> PhysicalOperator [dir=back];
        OME -> OpMember;
        OpMember -> MatchingCondition [dir=back];
    }

- :class:`~eko.evolution_operator.Operator`

    * represents a configuration for a fixed final scale :math:`Q_1^2`
    * performs the actual :doc:`computation </theory/DGLAP>`
    * uses the 3-dimensional :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis`
    * its :class:`~eko.member.OpMember` are only valid in the current
      threshold area

- :class:`~eko.evolution_operator.physical.PhysicalOperator`

    * is the connection of the :class:`~eko.evolution_operator.Operator`
      between the different flavor bases
    * is initialized with the 3-dimensional :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis`
    * does recombine the operator in the :ref:`theory/FlavorSpace:Operator Intrinsic QCD Evolution Basis`
      (see :doc:`Matching Conditions </theory/Matching>`)
    * exports the operators to :ref:`theory/FlavorSpace:Operator Flavor Basis` in a :class:`~numpy.ndarray`

- :class:`~eko.member.OpMember`

    * represents a single operator in Mellin space for a given element of the :ref:`theory/FlavorSpace:Operator Bases`
    * inside :class:`~eko.evolution_operator.Operator` they are in :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis`
    * inside :class:`~eko.evolution_operator.physical.PhysicalOperator` they are in :ref:`theory/FlavorSpace:Operator Intrinsic QCD Evolution Basis`
