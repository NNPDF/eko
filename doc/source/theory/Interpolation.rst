Interpolation
=============

Implementation: :mod:`eko.interpolation`

In order to obtain the operators in an PDF independent way we use approximation theory.
Therefore, we define the basis grid

.. math ::
    \mathbb G = \{ x_j : 0 < x_j <= 1, j=0,\ldots,N_{grid}-1 \}

from which we define our interpolation

.. math ::
    f(x) \sim \bar f(x) \sum\limits_{j=0}^{N_{grid} - 1 } f(x_j) p_j(x)

Thus each grid point :math:`x_j` has an associated interpolation polynomial :math:`p_j(x)`
(represented by :class:`eko.interpolation.BasisFunction`).
We interpolate in :math:`\ln(x)` using Lagrange interpolation among the nearest
:math:`N_{degree}+1` points, which renders the :math:`p_j(x)`: polynomials of order
:math:`O(\ln^{N_{degree}}(x))`.

Algorithm
---------

First, we split the interpolation region into several areas (represented by
:class:`eko.interpolation.Area`), which are bound by the grid points:

.. math ::
    A_j = (x_j,x_{j+1}], \quad \text{for}~j=0,\ldots,N_{grid}-2

Note, that we include the right border point into the definition, but not the left which
keeps all areas disjoint. This assumption is based on the physical fact, that PDFs do
have a fixed upper bound (:math:`x=1`), but no fixed lower bound.

Second, we define the interpolation blocks, which will build the interpolation polynomials
and contain the needed amount of points:

.. math ::
    B_j = \{x_j,\ldots,x_{j+N_{degree}+1}\} \quad \text{for}~j=0,\ldots,N_{grid}-N_{degree}-2

Now, we construct the interpolation in a bottom-up approach for a given point
:math:`\bar x` for the interpolating function :math:`\bar f(\bar x)`.
(The actual implementation is done in :class:`eko.interpolation.InterpolatorDispatcher`)

1. Determine the relevant active area :math:`\bar A(\bar x)` :

.. math ::
    \bar A(\bar x) = A_j : \bar x \in A_j

2. Determine the associated block :math:`\bar B(\bar A)` :
   Choose the block in which :math:`\bar A(\bar x)` is located most central.
   For an odd number of :math:`N_{degree}` choose the block which is
   located higher, i.e. closer to :math:`x=1` (following the choice of borders
   for the areas). At the border of the grid, choose the "most central".

3. Only the points and their associated polynomials which are in :math:`\bar B`
   participate in the interpolation of :math:`\bar f(\bar x)`.
   All other polynomials vanish:

.. math ::
    p_j(\bar x) = 0 \quad \text{for}~j : x_j \not\in \bar B

4. The active polynomials build a Lagrange interpolation over their associated points

.. math ::
    p_j(\bar x) = \prod\limits_{j\neq k} \frac{\ln(\bar x) - \ln(x_k)}{\ln(x_j) - \ln(x_k)} \quad \text{for}~j,k : x_{j,k} \in \bar B

The generated polynomials are orthonormal in the following sense

.. math ::
    p_j(x_k) = \delta_{jk}, \quad \text{for}~j,k=0,\ldots,N_{grid}-1

and thus continuous, however, they are not necessarily differentiable.
They are also complete in the following sense

.. math ::
    \sum\limits_{j=0}^{N_{grid}-1} p_j(x) = 1, \quad \text{for}~x \in (\text{min}(\mathbb G),\text{max}(\mathbb G)]


Example
-------
To outline the interpolation algorithm, let's consider the following example configuration:

.. figure :: ../img/interpolation-grid.png
    :align: center

    Example interpolation configuration with :math:`N_{grid}=9` points using a polynomial
    of degree :math:`N_{degree}=3` to interpolate. It has 8 areas and 6 blocks.

The mapping between areas and blocks are then given by

.. code-block:: none
   :caption: Mapping in the example configuration

   A0 -> B0, A1 -> B0, A2 -> B1, A3 -> B2, A4 -> B3, A5-> B4, A6 -> B5, A7 -> B5

The generated polynomials then look like:

.. figure :: ../img/interpolation-polynomials.png
    :align: center

    Selected polynomials of the example configuration. The 9 points have been placed
    logarithmically equidistant in :math:`(10^{-5},1]`. Note that :math:`p_4(x)`
    is still piecewise of degree 3.
