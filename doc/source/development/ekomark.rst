Ekomark
=======

Here we describe the design and API of the `ekomark` package.
The specific purpose of this package is to cointain all the utils to benchmark efficiently `eko`.
The underlying infrastructure is coming from `sqlite3` and `git-lfs` and it
is implemented in the package |banana|.

To install `ekomark` you can type:

``pip install ekomark``

.. important::

   Due to a problem in |banana| the only working version of ekomark can be insalled locally
   with:

   ``cd benchmarks && pip install -e .``

Among the external programs  olny |APFEL| provides a python wrapper, while |Pegasus|
bindings are available in: `N3PDF/external <https://github.com/N3PDF/external>`_.
No external program are needed to run the LHA benchmarks.


Ekomarl is composed by four subpackages:

* ``benchmark`` containing the runner, implementing the interface with the abstract class provided inside |banana| and the external utils that initialise and evolute the PDFs using the external programs.
* ``data`` which includes the module to generate `eko` like oprators cards and the module providing the operators database layout.
* ``navigator`` implementing the ekonavigator app.
* ``plot`` containing all the scipts to produce the output plots.


The banana configuration is loaded from ``banana_cfg.py`` file.
To run Ekomark see the section of the available :doc:`runners<ekomark_runners>`.
Furthermore Ekomark provides also a python interpter called `ekonavigator` to inspect
the cached benchmark reuslts.


.. toctree::
   :maxdepth: 1

   ekomark_runners.rst
   API <ekomark/ekomark.rst>
