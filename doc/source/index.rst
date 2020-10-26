Welcome to EKO!
===============

.. image:: img/Logo.png
  :width: 300
  :align: center
  :alt: EKO logo

EKO solves the |DGLAP| equations in Mellin space and produces evolution kernel operators (EKO).
It is |PDF|-independent and the operators can be computed for a given configuration
once and be reused after.

.. toctree::
    :maxdepth: 1
    :caption: Overview:
    :hidden:

    overview/features
    overview/examples
    overview/indices

.. toctree::
    :caption: Theory:
    :maxdepth: 1
    :hidden:

    theory/Interpolation
    theory/Mellin
    theory/pQCD
    theory/DGLAP
    theory/FlavourBasis
    theory/Matching
    
    zzz-refs

.. toctree::
    :maxdepth: 1
    :caption: Implementation:
    :hidden:

    code/IO
    API <modules/eko/eko>
    code/Operators
    code/Utilities
    code/Dependency

.. toctree::
    :maxdepth: 1
    :caption: Development:
    :hidden:
    
    development/Benchmarks
    ekomark <development/ekomark/ekomark>
    development/code_todos
