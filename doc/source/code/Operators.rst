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

        node [shape=box]
        OperatorGrid [label="OperatorGrid"];
        test, test1, test2222, test3 [style=invis];
        Operator [label="Operator" ];
        PhysicalOperator [label="PhysicalOperator"];
        OpMember [label="OpMember"];
        ndarray [label="np.ndarray"];

        {rank=same OperatorGrid test test1, test2222, test3}
        {rank=same Operator PhysicalOperator ndarray}

        OperatorGrid -> Operator [weight=1000];
        Operator -> OpMember;
        Operator -> PhysicalOperator [style=dashed len=10];
        PhysicalOperator -> ndarray [style=dashed len=10];
        PhysicalOperator -> OpMember
        OperatorGrid -> test -> test1 -> test2222 -> test3 [style=invis];
        test1 -> PhysicalOperator [weight=1000 style=invis];
        test3 -> ndarray [weight=1000 style=invis];
    }

- :class:`~eko.evolution_operator.grid.OperatorGrid`

    * this is the master class which administrates all operator tasks
    * it is instantiated once for each run
    * it holds all necessary :doc:`configurations </code/IO>`
    * it holds all necessary instances of the :doc:`/code/Utilities`

- :class:`~eko.evolution_operator.Operator`

    * this represents a configuration for a fixed final scale :math:`Q_1^2`
    * this performs the actual :doc:`computation </theory/DGLAP>`
    * this uses the 3-dimensional :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis`
    * its :class:`~eko.evolution_operator.member.OpMember` are only valid in the current
      threshold area

- :class:`~eko.evolution_operator.physical.PhysicalOperator`

    * this is the connection of the :class:`~eko.evolution_operator.Operator`
      between the different flavor bases
    * it is initialized with the 3-dimensional :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis`
    * it does recombine the operator in the :ref:`theory/FlavorSpace:Operator Evolution Basis`
      (see :doc:`Matching Conditions </theory/Matching>`)
    * it exports the operators to :ref:`theory/FlavorSpace:Operator Flavor Basis` in a :class:`~numpy.ndarray`

- :class:`~eko.evolution_operator.member.OpMember`

    * this represents a single operator in Mellin space for a given element of the :ref:`theory/FlavorSpace:Operator Bases`
    * inside :class:`~eko.evolution_operator.Operator` they are in :ref:`theory/FlavorSpace:Operator Anomalous Dimension Basis`
    * inside :class:`~eko.evolution_operator.physical.PhysicalOperator` they are in :ref:`theory/FlavorSpace:Operator Evolution Basis`