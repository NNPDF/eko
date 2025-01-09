Input & Output
==============

Input
-----

The input is split into two runcards: a **theory runcard** and an **operator runcard**.
Note that we are not assuming any default values for the fields, but instead the user has to provide
the full definition.

The theory card defines the general setup of the calculation, such as used perturbative orders or heavy quark masses.
Please see :class:`~eko.io.runcards.TheoryCard` for the actual format.

The observable card defines the specific setup of the EKOs, such as the target scales or the interpolation grid.
Please see :class:`~eko.io.runcards.OperatorCard` for the actual format.

The runcard objects are independent objects, but they can be parsed from a :py:obj:`dict` and, as usual, a :py:obj:`dict`
can be described, e.g., by a `yaml <https://github.com/yaml/pyyaml>`_ file.

Output
------

We output tar-compressed folder, which contains the used settings and the actual |EKO|.
To access the file you should use the interface provided by the :class:`~eko.io.struct.EKO` class and
you can follow :doc:`the tutorial </overview/tutorials/output>` to see how.

Each |EKO| is a rank-4 tensor with the indices ordered in the following way: ``EKO[pid_out][x_out][pid_in][x_in]``
where ``pid_out`` and ``x_out`` refer to the |PID| and the grid point
of the outgoing |PDF| and ``pid_in`` and ``x_in`` to the incoming |PDF|.
