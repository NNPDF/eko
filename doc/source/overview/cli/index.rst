Command-Line Interface
======================

EKO exposes also a CLI for running basic calculations and managing assets.

All the functions available to the CLI are also available in the Python package,
so for more intensive or repeated calculations the user is encouraged to write
his own scripts in Python, where all EKO functions are available, possibly using
``ekobox``, providing an additional UI layer.

The CLI is automatically installed together with the Python package, and it has
the same dependencies of ``ekobox`` (thus remember to install the package extra
``box``).

Subcommands
-----------

The following subcommands are currently available:

|command_run|_
   Fully managed EKO calculation.

|command_runcards|_
   Generate and manage runcards

|command_inspect|_
   Inspect the content of EKO files

|command_convert|_
   Convert an old EKO file to current format

.. |command_run| replace:: ``run``
.. _command_run: run.rst
.. |command_runcards| replace:: ``runcards``
.. _command_runcards: runcards.rst
.. |command_inspect| replace:: ``inspect``
.. _command_inspect: inspect.rst
.. |command_convert| replace:: ``convert``
.. _command_convert: convert.rst

For detailed information about command structure and flags query the command
help itself::

   eko --help
   # or individual subcommands
   eko inspect --help
