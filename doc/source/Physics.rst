Physics documentation
======================

Mathematical ingredients
------------------------

We solve the equations in Mellin-space as there multiplicative convolution is
mapped onto a normal multiplication and thus integro-differential equations,
such as DGLAP equations, are instead just normal differential equations.
The Mellin transformation is given by

.. math::
    \tilde g(N) = \mathcal{M}[g(x)](N) = \int\limits_{0}^{1} x^{N-1} g(x)\,dx

where we will denote all objects in Mellin-space with an additional tilde.
The inverse Mellin transformation is given by

.. math::
    g(x) = \mathcal{M}^{-1}[\tilde g(N)](x) = \frac{1}{2\pi i} \int\limits_{\mathcal{P}} x^{-N} \tilde g(N)\,dN

for a suitable path :math:`\mathcal{P}(t)`. The textbook path is given by
:math:`p(t) = c + i t`. In the code the more efficient Talbot
path :cite:`Abate` is used.

We will also use approximation theory, i.e., we interpolate the unkown PDF as a
weighted sum of polynomials

.. math::
    f(x,t_0) = \sum_{j=1}^{N_{grid}} f(x_j,t_0) \cdot p_j(x)

The current implementation uses a grid :math:`\{x_j\}` defined by the user and
a Lagrange interpolation in :math:`\log(x)` using the nearest :math:`k` points.
Thus the interpolation polynomials :math:`\{p_j(x)\}` are all of degree
:math:`(k-1)`.

The multiplicative convolution integral runs from :math:`x` to 1, thus only
basis functions which have support above :math:`x` may contribute to the
integral. This information is encoded in N-space in the following way: Due
to the Mellin kernel :math:`x^{N-1}` any piecewise polynomial, such as we
are doing, are proportional to
:math:`x_{\text{min/max}}^N = \exp(N\ln(x_{\text{min/max}}))`. When
multiplied with the inverse Mellin kernel :math:`x^{-N}` this leads to an
expresion which is propotional to

.. math::
    \exp(N(\ln(x_{\text{min/max}})-\ln(x)))\,.

Thus, if

.. math::
    \ln(x_{\text{min/max}})-\ln(x) <= 0 \Leftrightarrow x_{\text{min/max}} - x <= 0

we can integrate these terms with their own contour. We can choose the textbook
contour with :math:`Re(\mathcal P(t)) \to \infty` and thus these terms get
exponentially suppressed. In the limit these terms vanish exactly and thus we
exclude them from our numerical kernels.

QCD ingredients
---------------

We use perturbative QCD with the running coupling
:math:`a_s(\mu_F^2) = \alpha_s(\mu_F^2)/(4\pi)` given by
:cite:`Herzog:2017ohr` :cite:`Luthe:2016ima` :cite:`Baikov:2016tgj`

.. math::
      \frac{da_s}{d\ln\mu^2} = \beta(a) \
      = - \sum\limits_{n=0} \beta_n a_s^{n+2}


The Altarelli-Parisi splitting kernels can be expanded in powers of the strong
coupling :math:`a_s(\mu^2)` and are given by
:cite:`Moch:2004pa` :cite:`Vogt:2004mw`

.. math::
    \mathbf{P}(x,\mu^2)  = - \sum\limits_{n=0} a_s^{n+1}(\mu^2) \mathbf P^{(n)}(x)


Solving DGLAP
-------------

We are solving the DGLAP equations given by

.. math::
    \frac{d}{d\ln(\mu_F^2)} \mathbf{f}(x,\mu_F^2) =
        \int\limits_x^1\!\frac{dy}{y}\, \mathbf{P}(x/y) \cdot \mathbf{f}(y,\mu_F^2)


The non-siglet case can be solved in a linear way:

.. math::
    \frac{d f_{ns}(x,\mu_F^2)}{d\ln(\mu_F^2)} =
        \int\limits_x^1\!\frac{dy}{y}\, P_{ns}(y/x) \cdot f_{ns}(y,\mu_F^2)

We can thus write the non-siglet equations in N-space by

.. math::
    \frac{d\tilde f_{ns}(N,\mu_F^2)}{d\ln(\mu_F^2)} = \tilde P_{ns}(N) \cdot \tilde f_{ns}(N,\mu_F^2)

We also change the evolution variable to
:math:`t(\mu_F^2) = \ln(1/a_s(\mu_F^2))`
and then the differential equations to solve is given by

.. math::
    \frac{d\tilde f_{ns}(N,\mu_F^2)}{dt}
        = \frac{d\ln(\mu_F^2)}{dt} \cdot \frac{d\tilde f(N,\mu_F^2)}{d\ln(\mu_F^2)}
        = - \frac{a_s(\mu_F^2)}{\beta(a_s(\mu_F^2))} \tilde P_{ns}(N) \cdot \tilde f_{ns}(N,\mu_F^2)

Expanding the rhs to LO we get the final equation

.. math::
    \frac{d\tilde f^{(0)}(N,\mu_F^2)}{dt} = \frac{1}{\beta_0} \cdot \tilde P_{ns}^{(0)}(N) \cdot \tilde f_{ns}^{(0)}(N,\mu_F^2)

which is solved by

.. math::
    \tilde f^{(0)}(N,t_1) = \exp((t_1-t_0) \tilde P_{ns}^{(0)}(N)/\beta_0 ) \cdot \tilde f_{ns}^{(0)}(N,t_0)

Using the interpolation basis on the inital state PDF, we can define the
evolution kernel operator :math:`\hat O` by

.. math::
    \hat O_{k,j}^{ns,(0)}(t_1,t_0) = \mathcal{M}^{-1}\left[\exp((t_1-t_0)\tilde P_{ns}^{(0)}(N)/\beta_0)\tilde p_j(N)\right](x_k')

Now, we can write the solution to DGLAP in a true matrix operator scheme and
find

.. math::
    f^{(0)}(x_k,t_1) = \hat O_{k,j}^{(0)}(t_1,t_0) f^{(0)}(x_j,t_0)

The benchmarking LHA reference is given by :cite:`Giele:2002hx`
and :cite:`Dittmar:2005ed`.

References
----------

.. in order for the bibliography to work properly we need to generate _all_ references
    here (which then will link to here) - otherwise we may
    need to find out whether we can split the references into several
    files potentially ...

.. bibliography:: refs.bib
