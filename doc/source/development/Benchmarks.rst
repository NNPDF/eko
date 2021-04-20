Benchmarks
==========

|EKO| benchmarks are listed in the table below and are implemented in a separated tool :doc:`ekomark<ekomark>`.
For each external program the evolution can be perfomed at |LO|, |NLO|, |NNLO|. 

.. list-table:: Available Benchmarks
  :header-rows: 1

  * - Name
    - |FNS|
    - Scale Variations
    - Intrinsic evolution
    - Method
  * - LHA
    - VFNS, FFNS (nf=4)
    -
    -
    - ``iterate-exact``
  * - |lhapdf|
    -  FFNS (nf=4)
    -
    -
    - ``iterate-exact``
  * - |APFEL|
    - VFNS, FFNS
    - |T|
    - |T|
    - ``iterate-exact``, ``iterate-expanded``, ``truncated``
  * - |Pegasus|
    - VFNS, FFNS
    - |T|
    - 
    - ``iterate-exact``, ``iterate-expanded``, ``ordered-truncated``, ``truncated``


Les Houches Benchmarks
----------------------

The benchmarking LHA reference is given by :cite:`Giele:2002hx` (|LO| and |NLO|) and :cite:`Dittmar:2005ed` (|NNLO|).


List of bugs in :cite:`Giele:2002hx`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :math:`L_+ = 2(\bar u + \bar d)`
- head of table 1: :math:`\alpha_s(\mu_R^2 = 10^4~\mathrm{GeV}^2)=0.117574` (FFN) - as pointed out by :cite:`Dittmar:2005ed`
- in table 3, part 3: :math:`xL_-(x=10^{-4}, \mu_F^2 = 10^4~\mathrm{GeV}^2)=1.0121\cdot 10^{-4}` (wrong exponent) and
  :math:`xL_-(x=.1, \mu_F^2 = 10^4~\mathrm{GeV}^2)=9.8435\cdot 10^{-3}` (wrong exponent)

List of bugs in :cite:`Dittmar:2005ed`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- in table 15, part 1: :math:`xd_v(x=10^{-4}, \mu_F^2 = 10^4~\mathrm{GeV}^2) = 1.0699\cdot 10^{-4}` (wrong exponent) and
  :math:`xg(x=10^{-4}, \mu_F^2 = 10^4~\mathrm{GeV}^2) = 9.9694\cdot 10^{2}` (wrong exponent)

Lhapdf
------

|lhapdf| is the standard tool to store PDFs in Particle Physics. 
It provides a PDF dependent evolution method which can be compared with |Eko| applied to the same initial PDF. 

APFEL
-----

|APFEL| is a tool aimed to the evolution of PDFs and DIS observables' calculation
(and FTDY as well).
It has been used by the NNPDF collaboration up to NNPDF4.0

|APFEL| solves |DGLAP| numerically in x-space up to |NNLO|. QED evolution is also available.  
The programs provides 3 different strategies, and in various theory setups (|FNS|, SV, IC ) as shown in the table.
As |Eko|, |APFEL| can be interfaced with |lhapdf|.

Pegasus
-------

|Pegasus| is a tool aimed exclusively to the evolution of PDFs, it is written in Fortran.
This program has been used to produce the LHA tables.

|Pegasus| solves |DGLAP| numerically in N-space up to |NNLO|.
The programs provides 3 different strategies, with various |FNS| and  Scale Variations as shown in the table.
|Pegasus| takes as input a pdf with a fixed funtional form and it's not interfaced with |lhapdf|.
Aslo the starting scale must be equal to the scale at which the reference value of :math:`\alpha_s` is provided.
