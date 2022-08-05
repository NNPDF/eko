Input & Output
==============

We use `yaml <https://github.com/yaml/pyyaml>`_ runcards for piping input and output files.

Input
-----

The input is split into two runcards:

- a **theory runcard** (compatible with the NNPDF theory database) that defines the physical setup
  and environment. The full list of available entries is described in the following table.
  None of the keyword has a default value, however if you want to use a default runcard you can import the one
  provided in :mod:`banana.data.theories.default_card` and update only the needed entries.

  .. list-table:: theory input runcard
    :header-rows: 1

    * - Name
      - Type
      - description
      - optional
    * - ``PTO``
      - :py:obj:`int`
      - |QCD| perturbation theory order: ``0`` = LO, ``1`` = NLO, ``2`` = NNLO, ``3`` = N3LO.
      -
    * - ``ModEv``
      - :py:obj:`str`
      - Evolution method. Possible options are:
        ``iterate-exact`` abbreviated with ``EXA``, ``decompose-exact``, ``perturbative-exact``,
        ``iterate-expanded`` abbreviated with ``EXP``, ``decompose-expanded``, ``perturbative-expanded``,
        ``truncated`` abbreviated with ``TRN``, ``ordered-truncated``.
      -
    * - ``fact_to_ren_scale_ratio``
      - :py:obj:`float`
      - Factorization to renormalization scale ratio.
      -
    * - ``ModSV``
      - :py:obj:`str`
      - Scale variation method, used only if ``fact_to_ren_scale_ratio!=1``. Possible options are:
        ``expanded`` or `exponentiated``.
      - |T|
    * - ``Q0``
      - :py:obj:`float`
      - Initial |PDF| evolution scale (in GeV).
      -
    * - ``nf0``
      - :py:obj:`int`
      - Number of flavors active ant the ``Q0`` scale.
        If not provided it is inferred from the heavy quarks threshold scales.
      - |T|
    * - ``MaxNfPdf``
      - :py:obj:`int`
      - Maximum number of flavors in the |PDF| evolution.
      -
    * - ``alphas``
      - :py:obj:`float`
      - Reference value of the strong coupling :math:`\alpha_s` (Note that we have to use
        :math:`\alpha_s` here, instead of :math:`a_s` for legacy reasons).
      -
    * - ``Qref``
      - :py:obj:`float`
      - Reference scale at which the ``alphas`` value is given (in GeV).
      -
    * - ``nfref``
      - :py:obj:`int`
      - Number of flavors active ant the ``Qref`` scale.
        If not provided it is inferred from the heavy quarks threshold scales.
      - |T|
    * - ``MaxNfAs``
      - :py:obj:`int`
      - Maximum number of flavors in the strong coupling evolution.
      -
    * - ``QED``
      - :py:obj:`bool`
      - If ``1`` include |QED| evolution.
      -
    * - ``alphaqed``
      - :py:obj:`float`
      - Reference value of the electromagnetic coupling :math:`\alpha_em`.
      -
    * - ``Qedref``
      - :py:obj:`float`
      - Reference scale at which the ``alphaqed`` value is given (in GeV).
      -
    * - ``HQ``
      - :py:obj:`str`
      - Heavy quark scheme: "POLE" = use heavy quark pole masses, "MSBAR" = use heavy quarks :math:`\overline_{MS}` masses.
      -
    * - ``mc``
      - :py:obj:`float`
      - Charm quark mass (in GeV).
      -
    * - ``Qmc``
      - :py:obj:`float`
      - Reference scale at which the charm quark mass is given (in GeV). Used only with ``HQ='MSBAR'``.
      - |T|
    * - ``kcThr``
      - :py:obj:`float`
      - Ratio between the charm mass scale and the ``nf=4`` threshold scale.
      -
    * - ``mb``
      - :py:obj:`float`
      - Bottom quark mass (in GeV).
      -
    * - ``Qmb``
      - :py:obj:`float`
      - Reference scale at which the bottom quark mass is given (in GeV). Used only with ``HQ='MSBAR'``.
      - |T|
    * - ``kbThr``
      - :py:obj:`float`
      - Ratio between the bottom mass scale and the ``nf=5`` threshold scale.
      -
    * - ``mt``
      - :py:obj:`float`
      - Top quark mass (in GeV).
      -
    * - ``Qmt``
      - :py:obj:`float`
      - Reference scale at which the top quark mass is given (in GeV). Used only with ``HQ='MSBAR'``.
      - |T|
    * - ``ktThr``
      - :py:obj:`float`
      - Ratio between the top mass scale and the ``nf=6`` threshold scale.
      -
    * - ``IC``
      - :py:obj:`bool`
      - If ``1`` allow for intrinsic charm evolution.
      -
    * - ``IB``
      - :py:obj:`bool`
      - If ``1`` allow for intrinsic bottom evolution.
      -
    * - ``ID``
      - :py:obj:`int`
      - Theory identifier, see NNPDF conventions.
      - |T|


- an **operator runcard** that defines the numerical setup and the requested operators.
  The full list of available entries is described in the following table.
  Also here none of the keyword has a default value, however default runcard is provided in
  :mod:`ekomark.data.operators`.


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
      - Q2-grid at which the |EKO| is computed (in GeV squared).
    * - ``interpolation_polynomial_degree``
      - :py:obj:`int`
      - Maximum interpolation polynomial degree.
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

The :class:`~eko.output.Output` instance represents the following runcard:

.. list-table:: output runcard
  :header-rows: 1

  * - Name
    - Type
    - description
  * - ``interpolation_xgrid``
    - :py:obj:`list(float)`
    - the interpolation grid
  * - ``interpolation_polynomial_degree``
    - :py:obj:`int`
    - polynomial degree of the interpolating function
  * - ``interpolation_is_log``
    - :py:obj:`bool`
    - use logarithmic interpolation?
  * - ``q2_ref``
    - :py:obj:`float`
    - starting scale
  * - ``pids``
    - :py:obj:`list(int)`
    - participating partons listed by their PDG id
  * - ``Q2grid``
    - :py:obj:`dict`
    - all operators at the requested values of :math:`Q^2` represented by the key

The grid elements contains two keys each

- ``operators`` a :py:obj:`dict` with all evolution kernel operators where the key indicates which distribution is generated by which other one
  and the value represents the eko in matrix representation - this can either be the plain list representation or the binary representation
  (as provided by :py:meth:`numpy.ndarray.tobytes`)
- ``operator_errors`` a :py:obj:`dict` with the integration errors associated to the respective operators following the same conventions as
  the ``operator`` dictionary
- each element (|EKO|) is a rank-4 tensor with the indices ordered in the following way: ``EKO[pid_out][x_out][pid_in][x_in]`` where ``pid_out`` and ``x_out``
  refer to the outgoing |PDF| and ``pid_in`` and ``x_in`` to the incoming |PDF|. The ordering of ``pid_out/pid_in`` is determined by the ``pids``
  parameter of the output and the order of ``x_out/x_in`` by ``interpolation_xgrid``.
