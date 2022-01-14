Mellin Space and Transformations
================================

We solve the equations in Mellin-space as there multiplicative convolution is
mapped onto a normal multiplication and thus integro-differential equations,
such as |DGLAP| equations, are instead just normal differential equations.

The Mellin transformation is given by

.. math::
    \tilde g(N) = \mathcal{M}[g(x)](N) = \int\limits_{0}^{1} x^{N-1} g(x)\,dx

We will denote objects in Mellin-space with an additional tilde if they may appear in
both spaces. The inverse Mellin transformation is given by

.. math::
    g(x) = \mathcal{M}^{-1}[\tilde g(N)](x) = \frac{1}{2\pi i} \int\limits_{\mathcal{P}} x^{-N} \tilde g(N)\,dN

for a suitable path :math:`\mathcal{P}(t)` which runs to the right to the right-most
pole of :math:`\tilde g(N)`. For the implementation we will assume that the integration path
:math:`\mathcal P` is given by

.. math::
    \mathcal P : [0,1] \to \mathbb C : t \to \mathcal P(t)\quad
        \text{with}~\mathcal P(1/2-t) = \mathcal P^*(1/2+t)

where :math:`\mathcal P^*` denotes complex conjugation. Assuming further :math:`\tilde g`
to be a holomorphic function along the path, we can rewrite the inversion integral by

.. math::

    g(x) &= \frac{1}{2\pi i} \int\limits_{0}^{1} x^{-\mathcal{P}(t)} \tilde g(\mathcal{P}(t)) \frac{d\mathcal{P}(t)}{dt} \,dt\\
         &= \frac{1}{\pi} \int\limits_{1/2}^{1} \Re \left(  x^{-\mathcal{P}(t)} \tilde g(\mathcal{P}(t)) \frac 1 i \frac{d\mathcal{P}(t)}{dt} \right) \,dt


Important Examples
------------------

Polynomials
^^^^^^^^^^^

.. math ::
    \mathcal M[x^m](N) &= \frac 1 {N + m}\\
    \mathcal M[x^m\Theta(x - x_{min})\Theta(x_{max} - x)](N) &= \frac {x_{max}^{N+m} - x_{min}^{N+m}} {N + m}

Logarithms
^^^^^^^^^^

.. math ::
    \mathcal M[\ln(x)^m](N) &= \frac{d^m}{dN^m}\frac 1 {N}\\
    \mathcal M[\ln^m(x)\Theta(x - x_{min})\Theta(x_{max} - x)](N) &= \frac{d^m}{dN^m}\frac {x_{max}^{N} - x_{min}^{N}} {N}

Note that any derivative to either :math:`x_m^N` or :math:`1/N` is again proportional to its source.

Plus Distributions
^^^^^^^^^^^^^^^^^^

.. math ::
    \mathcal M[1/(1-x)_+](N) = S_1(N)

with the harmonic sum :math:`S_1` (see :ref:`theory/mellin:harmonic sums`).


Inversion of Factorizable Kernels
---------------------------------

If the integration kernel :math:`\tilde g(N)` can be factorized

.. math::
    \tilde g(N) = x_0^N \cdot \tilde h(N)

with :math:`x_0` a fixed number in :math:`(0,1]` and :math:`\lim_{N\to\infty}h(N)\to 0`,
the inversion can be simplified if the inversion point :math:`x_i` is **above** :math:`x_0`.

.. math::
    g(x_i) &= \frac{1}{2\pi i} \int\limits_{\mathcal{P}} x_i^{-N} x_0^N \tilde h(N)\,dN \\
           &= \frac{1}{2\pi i} \int\limits_{\mathcal{P}} \exp(-N(\ln(x_i)-\ln(x_0))) \tilde h(N)\,dN

Now, take the textbook path :math:`p : \mathbb R \to \mathbb C : t \to p(t) = c + i t` and
consider the limit in which we shift the parameter :math:`c \to \infty`.
As :math:`x_i > x_0` it follows immediately
:math:`\ln(x_i)-\ln(x_0) > 0` and thus

.. math::
    |\exp(-N(\ln(x_i)-\ln(x_0)))| \to 0

Together with the assumed vanishing of :math:`\tilde h(N)` we can conclude
:math:`g(x_i) = 0`.


Convolution
-----------

Mellin space factorizes multiplicative convolution

.. math ::
    (f \otimes g)(x) &= \int\limits_x^1 \frac{dy}{y} f(x/y) g(y)\\
    \mathcal M[(f \otimes g)(x)](N) &= \mathcal M[f(x)](N) \cdot \mathcal M[g(x)](N)

The convolution integral runs from :math:`x` to 1, thus only
basis functions which have support above :math:`x` may contribute to the
integral. This information is encoded in N-space in the following way: Due
to the Mellin kernel :math:`x^{N-1}` any piecewise polynomial, such as we
are doing, are proportional to
:math:`x_{\text{min/max}}^N = \exp(N\ln(x_{\text{min/max}}))`
(see :ref:`theory/mellin:important examples`). They are thus factorizable is the above sense.

Harmonic Sums
-------------
In the computations of the anomalous dimensions (generalized) harmonic sums
:cite:`Ablinger:2013hcp` appear naturally:

.. math ::
    S_{m}(N) &= \sum\limits_{j=1}^N \frac{(\text{sign}(m))^j}{j^{|m|}} \\
    S_{m_0,m_1\ldots}(N) &= \sum\limits_{j=1}^N \frac{(\text{sign}(m_0))^j}{j^{|m_0|}} S_{m_1\ldots}(j)

We then need to find an analytical continuation of these sums into the complex plain to perform
the Mellin inverse.

- the sums :math:`S_{m}(N)` for :math:`m > 0` do have a straight continuation:

.. math ::
    S_m(N) = \sum\limits_{j=1}^N \frac 1 {j^m} = \frac{(-1)^{m-1}}{(m-1)!} \psi_{m-1}(N+1)+c_m \quad
    \text{with}\, c_m = \left\{\begin{array}{ll} \gamma_E, & m=1\\ \zeta(m), & m>1\end{array} \right.

and where :math:`\psi_k(N)` is the :math:`k`-th polygamma function (implemented as :meth:`~eko.anomalous_dimensions.harmonics.cern_polygamma`)
and :math:`\zeta` the Riemann zeta function (using :func:`scipy.special.zeta`).

- for the sums :math:`S_{-m}(N)` and :math:`m > 0` we use :cite:`Gluck:1989ze`

.. math ::
    S_m'(N) = 2^{m-1}(S_m(N) + S_{-m}(N)) = \frac{1+\eta}{2} S_m(N/2) + \frac{1-\eta}{2}S_m((N-1)/2)

where formally :math:`\eta = (-1)^N` but in all singlet-like quantities it has to be analytically continued with 1
and with -1 elsewise.

- for the sums with greater depth we use the lists provided in :cite:`Gluck:1989ze,MuselliPhD,Blumlein:1998if`.
- For :math:`S_{-2,1}(N)` we use the implementation of :cite:`Gluck:1989ze` (where it is called :math:`\tilde S`):

.. math ::
    S_{-2,1}(N) &= - \frac 5 8 \zeta(3) + \zeta(2)\left(S_{-1}(N) - \frac{\eta}{N} + \log(2)\right) + \eta\left(\frac{S_{1}(N)}{N^2} + g_3(N)\right)\\
    g_3(N) &= \mathcal M \left[\frac{\text{Li}_2(x)}{1+x}\right](N)

where for :math:`g_3(N)` we use the parametrization of :cite:`Vogt:2004ns` (implemented as :meth:`~eko.anomalous_dimensions.harmonics.mellin_g3`).
