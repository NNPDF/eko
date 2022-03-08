Ekomark
=======

Here we describe the design and API of the `ekomark` package.
The specific purpose of this package is to contain all the utils to benchmark efficiently `eko`.
The underlying infrastructure is coming from `sqlite3` and `git-lfs` and it
is implemented in the package |banana|.

To install `ekomark` you can type:

``pip install ekomark``

.. important::

   Due to a problem in |banana| the only working version of ekomark can be installed locally
   with:

   ``cd benchmarks && pip install -e .``

Among the external programs  only |APFEL| provides a python wrapper, while |Pegasus|
bindings are available in: `N3PDF/external <https://github.com/N3PDF/external>`_.
No external program are needed to run the LHA benchmarks.


Ekomark is composed by four subpackages:

* ``benchmark`` containing the runner, implementing the interface with the abstract class provided inside |banana| and the external utils that initialize and evolute the PDFs using the external programs.
* ``data`` which includes the module to generate `eko` like operators cards and the module providing the operators database layout.
* ``navigator`` implementing the ekonavigator app.
* ``plot`` containing all the scripts to produce the output plots.


The banana configuration is loaded from ``banana/cfg.py`` file.
To run Ekomark see the section of the available :doc:`runners<ekomark_runners>`.
Furthermore Ekomark provides also a python interpreter called `ekonavigator` to inspect
the cached benchmark results.


.. toctree::
   :maxdepth: 1

   ekomark_runners.rst
   API <ekomark/ekomark.rst>
