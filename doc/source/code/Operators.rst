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
        Operator [label="Operator" ];
        PhysicalOperator [label="PhysicalOperator"];
        OpMember [label="OpMember"];

        OperatorGrid -> Operator;
        Operator -> OpMember;
        Operator -> PhysicalOperator [style=dashed];
    }

- :class:`~eko.operator.grid.OperatorGrid`

    *  this is the master class which administrates all operator tasks
    *  it is instantiated once for each run
    *  it divides the given range of :math:`Q^2` into the necessary threshold crossings

- :class:`~eko.operator.Operator`

    * this represents a configuration for a fixed final scale :math:`Q^2`
    * this uses the 3-dimensional anomalous dimension basis
    * its :class:`~eko.operator.member.OpMember` are only valid in the current
      threshold area

- :class:`~eko.operator.physical.PhysicalOperator`

    * this is the connection of the :class:`~eko.operator.Operator`
      between the different flavor bases

- :class:`~eko.operator.member.OpMember`

    * this represents a single operator in Mellin space for a fixed flavor space operator
    * inside :class:`~eko.operator.Operator` they are in anomalous dimension basis, i.e.
      :math:`\tilde{\mathbf{E}}_S, \tilde{E}_{ns}^{\pm,v}`, and they never cross a threshold
    * inside :class:`~eko.operator.physical.PhysicalOperator` they are in evolution
      basis, i.e. they evolve e.g. :math:`\tilde V, \tilde T_3` etc., so they are eventually
      a product of the anomalous dimension basis (see :doc:`Matching Conditions </theory/Matching>`)