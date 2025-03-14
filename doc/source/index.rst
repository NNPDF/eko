Welcome to EKO!
===============

.. image:: img/Logo.png
  :width: 300
  :align: center
  :alt: EKO logo

EKO solves the |DGLAP| equations in Mellin space and produces evolution kernel operators (EKO).
It is |PDF|-independent and the operators can be precomputed for a given configuration
once and be reused after.

EKO is ...

- open-source since the beginning - allowing a community effort for writing a new generation of code
- written in Python - opting for a popular, high-level langauge to facilitate other authors to participate in the project
- equipped with a continous integration / deployment - to allow a high coding standard and routinely checked code basis
- part of the N3PDF software compendium: |yadism|, |banana|, |pineappl| and |pineko|

.. toctree::
    :maxdepth: 1
    :caption: Overview:
    :hidden:

    overview/features
    overview/tutorials/index
    CLI <overview/cli/index>
    overview/indices

.. toctree::
    :caption: Theory:
    :maxdepth: 1
    :hidden:

    theory/Interpolation
    theory/Mellin
    theory/FlavorSpace
    theory/pQCD
    theory/DGLAP
    theory/N3LO_ad
    theory/Matching
    theory/TimeLike
    theory/MHOU

    zzz-refs

.. toctree::
    :maxdepth: 1
    :caption: Implementation:
    :hidden:

    code/IO
    code/interface
    API <modules/eko/eko>
    Math <modules/ekore/ekore>
    code/Operators
    code/Utilities

.. toctree::
    :maxdepth: 1
    :caption: Development:
    :hidden:

    development/Benchmarks.rst
    development/ekomark.rst
