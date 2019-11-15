
Physics documentation
=====================

Mathematical ingredients
------------------------

We solve the equations in Mellin-space as there multiplicative convolution is
mapped onto a normal multiplication and thus the integro-differential equations
are instead just normal differential equations.
The Mellin transformation is given by

.. math::
    \tilde g(N) = \mathcal{M}[g(x)](N) = \int\limits_{0}^{1} x^{N-1} g(x)\,dx

where we will denote all objects in Mellin-space with an additional tilde.
The inverse Mellin transformation is given by

.. math::
    g(x) = \mathcal{M}^{-1}[\tilde g(N)](x) = \frac{1}{2\pi i} \int\limits_{c-i\infty}^{c+i\infty} x^{-N} \tilde g(N)\,dN

QCD ingredients
---------------


The running coupling :math:`a_s(\mu_F^2) = \alpha_s(\mu_F^2)/(4\pi)`
is given by :cite:`Herzog:2017ohr` :cite:`Luthe:2016ima` :cite:`Baikov:2016tgj`

.. math::
      \frac{da_s}{d\ln\mu^2} = \beta(a) \
      = - \sum\limits_{n=0} \beta_n a_s^{n+2}


The Altarelli-Parisi splitting kernels can be expanded in powers of the strong
coupling :math:`a_s(\mu_F^2)` given by :cite:`Moch:2004pa` :cite:`Vogt:2004mw`

.. math::
    \mathbf{P}(x)
        = - \sum\limits_{n=0} a_s^{n+1} \mathbf P^{(n)}(x)


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
and differential equations to solve is given by

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

We will now assume further, that we can write the initial state as
a superposition of weighted polynomials:

.. math::
    f(x,t_0) = \sum_{j=1}^{N_{grid}} f(x_j,t_0) \cdot p_j(x)

We can thus define our EKO :math:`\hat O` by

.. math::
    \hat O_{k,j}^{(0)}(t_1,t_0) = \mathcal{M}^{-1}\left[\exp((t_1-t_0)\tilde P_{ns}^{(0)}(N)/\beta_0)\tilde p_j(N)\right](x_k')

where the grid points :math:`\{x_k'\}` do not necessarily have to
coincident with the interpolation grid :math:`\{x_j\}`. Now, we have

.. math::
    f^{(0)}(x_k,t_1) = \hat O_{k,j}^{(0)}(t_1,t_0) f^{(0)}(x_j,t_0)

The benchmarking LHA reference is given by :cite:`Giele:2002hx`.

References
----------

.. in order for the bibliography to work properly we need to generate _all_ references
    here (which then will link to here) - otherwise we may
    need to find out whether we can split the references into several
    files potentially ...

.. bibliography:: refs.bib
