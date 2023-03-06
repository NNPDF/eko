Input & Output
==============

We use `yaml <https://github.com/yaml/pyyaml>`_ runcards for piping input and output files.

Input
-----

The input is split into two runcards: a **theory runcard** and an **operator runcard**.
Note that we are not assuming any default values for the keywords, but instead the user has to provide
the full definition. However, for the :doc:`benchmarking environment </development/Benchmarks>` we do provide
some default settings.

Theory Runcard
^^^^^^^^^^^^^^

The **theory runcard** (compatible with the NNPDF theory database) defines the physical setup
and environment. The benchmark settings are available at :mod:`banana.data.theories.default_card`.

.. list-table:: theory input runcard
  :header-rows: 1

  * - Name
    - Type
    - Description
  * - ``PTO``
    - :py:obj:`int`
    - |QCD| perturbation theory order: ``0`` = |LO|, ``1`` = |NLO|, ``2`` = |NNLO|, ``3`` = |N3LO|.
  * - ``ModEv``
    - :py:obj:`str`
    - Evolution method. Possible options are:
      ``iterate-exact`` abbreviated with ``EXA``, ``decompose-exact``, ``perturbative-exact``,
      ``iterate-expanded`` abbreviated with ``EXP``, ``decompose-expanded``, ``perturbative-expanded``,
      ``truncated`` abbreviated with ``TRN``, ``ordered-truncated``.
  * - ``XIF``
    - :py:obj:`float`
    - Factorization to renormalization scale ratio. ``1`` means no scale variation.
  * - ``ModSV``
    - :py:obj:`str`
    - Scale variation method, used only if ``XIF!=1``. Possible options are:
      ``expanded`` or ``exponentiated``.
  * - ``Q0``
    - :py:obj:`float`
    - Initial |PDF| evolution scale (in GeV).
  * - ``nf0``
    - :py:obj:`int` or :py:obj:`None`
    - Number of flavors active ant the ``Q0`` scale.
      If not provided it is inferred from the heavy quark threshold scales.
  * - ``MaxNfPdf``
    - :py:obj:`int`
    - Maximum number of flavors in the |PDF| evolution.
  * - ``alphas``
    - :py:obj:`float`
    - Reference value of the strong coupling :math:`\alpha_s` (Note that we have to use
      :math:`\alpha_s` here, instead of :math:`a_s` for legacy reasons).
  * - ``Qref``
    - :py:obj:`float`
    - Reference scale at which the ``alphas`` value is given (in GeV).
  * - ``nfref``
    - :py:obj:`int` or :py:obj:`None`
    - Number of flavors active at the ``Qref`` scale.
      If not provided it is inferred from the heavy quark threshold scales.
  * - ``MaxNfAs``
    - :py:obj:`int`
    - Maximum number of flavors in the strong coupling evolution.
  * - ``QED``
    - :py:obj:`int`
    - If ``1`` include |QED| evolution.
  * - ``alphaqed``
    - :py:obj:`float`
    - Reference value of the electromagnetic coupling :math:`\alpha_{em}`.
  * - ``Qedref``
    - :py:obj:`float`
    - Reference scale at which the ``alphaqed`` value is given (in GeV).
  * - ``HQ``
    - :py:obj:`str`
    - Heavy quark scheme: if ``POLE`` use heavy quark pole masses, if ``MSBAR`` use heavy quark |MSbar| masses.
  * - ``mc``
    - :py:obj:`float`
    - Charm quark mass (in GeV).
  * - ``Qmc``
    - :py:obj:`float`
    - Reference scale at which the charm quark mass is given (in GeV). Used only with ``HQ='MSBAR'``.
  * - ``kcThr``
    - :py:obj:`float`
    - Ratio between the charm mass scale and the ``nf=4`` threshold scale.
  * - ``mb``
    - :py:obj:`float`
    - Bottom quark mass (in GeV).
  * - ``Qmb``
    - :py:obj:`float`
    - Reference scale at which the bottom quark mass is given (in GeV). Used only with ``HQ='MSBAR'``.
  * - ``kbThr``
    - :py:obj:`float`
    - Ratio between the bottom mass scale and the ``nf=5`` threshold scale.
  * - ``mt``
    - :py:obj:`float`
    - Top quark mass (in GeV).
  * - ``Qmt``
    - :py:obj:`float`
    - Reference scale at which the top quark mass is given (in GeV). Used only with ``HQ='MSBAR'``.
  * - ``ktThr``
    - :py:obj:`float`
    - Ratio between the top mass scale and the ``nf=6`` threshold scale.
  * - ``IC``
    - :py:obj:`bool`
    - If ``1`` allow for intrinsic charm evolution.
  * - ``IB``
    - :py:obj:`bool`
    - If ``1`` allow for intrinsic bottom evolution.

Operator Runcard
^^^^^^^^^^^^^^^^


The **operator runcard** defines the numerical setup and the requested operators.
The benchmark settings are available at :mod:`ekomark.data.operators`.


.. list-table:: operator input runcard
  :header-rows: 1

  * - Name
    - Type
    - description
  * - ``interpolation_xgrid``
    - :py:obj:`list(float)`
    - x-grid at which the |EKO| is computed.
  * - ``Q2grid``
    - :py:obj:`list(float)`
    - Q2-grid at which the |EKO| is computed (in GeV^2).
  * - ``interpolation_is_log``
    - :py:obj:`bool`
    - If ``True`` use logarithmic interpolation.
  * - ``interpolation_polynomial_degree``
    - :py:obj:`int`
    - Polynomial degree of the interpolating function.
  * - ``debug_skip_non_singlet``
    - :py:obj:`bool`
    - If ``True`` skip the non singlet sector, useful for debug purposes.
  * - ``debug_skip_singlet``
    - :py:obj:`bool`
    - If ``True`` skip the singlet sector, useful for debug purposes.
  * - ``ev_op_max_order``
    - :py:obj:`int`
    - Perturbative expansion order of unitary evolution matrix.
      Needed only for ``perturbative`` evolution methods.
  * - ``ev_op_iterations``
    - :py:obj:`int`
    - Number of evolution steps.
  * - ``backward_inversion``
    - :py:obj:`str`
    - Backward matching inversion method, relevant only for backward evolution in |VFNS|.
  * - ``n_integration_cores``
    - :py:obj:`int`
    - Number of cores used during the integration. ``0`` means use all; ``-1`` all minus 1.

Output
------

The eko output is represented by the class :class:`~eko.output.Output`.
An instance of this class is a :py:obj:`dict` and contains the following keys:

.. list-table:: output runcard
  :header-rows: 1

  * - Name
    - Type
    - Description
  * - ``Q2grid``
    - :py:obj:`dict`
    - All operators at the requested values of :math:`Q^2` represented by the key
  * - ``eko_version``
    - :py:obj:`float`
    - The |EKO| version
  * - ``inputgrid``
    - :py:obj:`list(float)`
    - The input x-grid
  * - ``inputpids``
    - :py:obj:`list(int)`
    - The input list of participating partons listed by their |PID|.
  * - ``interpolation_is_log``
    - :py:obj:`bool`
    - If ``True`` use logarithmic interpolation.
  * - ``interpolation_polynomial_degree``
    - :py:obj:`int`
    - Polynomial degree of the interpolating function.
  * - ``targetgrid``
    - :py:obj:`list(float)`
    - The target x-grid
  * - ``targetpids``
    - :py:obj:`list(int)`
    - The target list of participating partons listed by their |PID|

Since the final |EKO| is a rank 4-tensor we store in the output all the different grids
for each dimension:``targetpids,targetgrid,inputpids,inputgrid``.
The ``Q2grid`` values are the actual tensor for the requested :math:`Q^2`. Each of them contains two keys:

- ``operators`` a :py:obj:`dict` with all evolution kernel operators where the key indicates which distribution is generated by which other one
  and the value represents the eko in matrix representation - this can either be the plain list representation or the binary representation
  (as provided by :py:meth:`numpy.ndarray.tobytes`)
- ``errors`` a :py:obj:`dict` with the integration errors associated to the respective operators following the same conventions as
  the ``operator`` dictionary

Each element (|EKO|) is a rank-4 tensor with the indices ordered in the following way: ``EKO[pid_out][x_out][pid_in][x_in]`` where ``pid_out`` and ``x_out``
refer to the outgoing |PDF| and ``pid_in`` and ``x_in`` to the incoming |PDF|. The ordering of ``pid_out/pid_in`` is determined by the ``targetpids/inputpids``
parameter of the output and the order of ``x_out/x_in`` by ``targetgrid/inputgrid``.

To further explore how an :class:`~eko.output.Output` object looks like
you can follow :doc:`this tutorial </overview/tutorials/output>`.
