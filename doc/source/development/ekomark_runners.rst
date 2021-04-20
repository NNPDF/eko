Usage
=====

Ekomark mimics the same inputs needed to run `eko`, namely a theory card, an 
observable card and also the name of a pdf set whenever the external program can be used 
together with |lhapdf|. 

Both the theory and observable card can be gnerated authomatically from a default: 
the former with |banana|, the latter with something similar to ``generate_observable()`` provided `sandbox.py`. 

In addition to run `ekomark` you need to specify the external program you would benchmark against. 
To do so, you will have to initialise a class of type ``ekomark.benchmark.runner``.
To speed up the calculations null PDFs can be skipped setting the attribute ``skip_pdfs``
Finally you can decide to display the output in Flavor or in Evolution basis setting ``rotate_to_evolution_basis``

In the following section we describe some available `runners` which are the most useful example.

The minimal setup of the input cards must contain: 

.. list-table:: minimal theory input runcard
  :header-rows: 1

  * - Name
    - Type
    - default
    - description
  * - ``PTO``
    - :py:obj:`int`
    - [required]
    - order of perturbation theory: ``0`` = LO, ...
  * - ``alphas``
    - :py:obj:`float`
    - [required]
    - reference value of the strong coupling :math:`\alpha_s(\mu_0^2)` (Note that we have to use
      :math:`\alpha_s(\mu_0^2)` here, instead of :math:`a_s(\mu_0^2)` for legacy reasons)
  * - ``Q0``
    - :py:obj:`float`
    - [required]
    - reference scale from which to start
  * - ``mc``
    - :py:obj:`float`
    - 2.0
    - charm mass in GeV
  * - ``mb``
    - :py:obj:`float`
    - 4.5
    - bottom mass in GeV
  * - ``mt``
    - :py:obj:`float`
    - 173.0
    - top mass in GeV


.. list-table:: minimal oprator input runcard
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
  * - ``Q2grid``
    - :py:obj:`dict`
    - all operators at the requested values of :math:`Q^2` represented by the key


The output of `ekomark` will be stored in ``data/benchmark.db`` inside a :py:obj:`Pandas.DataFrame` table.
You can then use the `navigator` app to inspect your database and produce plots.

Available Runners
-----------------

In ``benchmarks/runners`` we provide a list of established benchmarks

- ``sandbox.py``:

  - it is used to provide the boilerplate needed for a basic run,
    in order to make a quick run for debugging purpose, but still fully managed
    and registered by the `ekomark` machinery and then available in the
    `navigator`

- ``apfel_bench.py``:

  - it is used by the corresponding workflow to
    run the established benchmarks against |APFEL|. The complete 
    run of this script will benchmark |EKO| against all the compatible |APFEL| features. 
  - the necessary python bindings are provided by the |APFEL| itself

- ``pegaus_bench.py``:

  - it is used by the corresponding workflow to
    run the established benchmarks against |Pegasus|. The complete 
    run of this script will benchmark |EKO| against all the compatible |Pegasus| features. 
  - the necessary python bindings are provided by us externally.

- ``paper_LHA_bench.py``:

  - it is used by the corresponding workflow to
    run the established benchmarks against the LHA papers. 
  - There are no external python bindings needed since the LHA data are stored in 
    ``ekomark/benchmark/external/LHA.yaml``.

All of them are examples useful to understand how to use the 
`ekomark` package for benchmarking.
