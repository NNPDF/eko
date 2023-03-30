Solving DGLAP
=============

We are solving the |DGLAP| equations :cite:`Altarelli:1977zs,Dokshitzer:1977sg,Gribov:1972ri` given in x-space by

.. math::
    \frac{d}{d\ln(\mu_F^2)} \mathbf{f}(x,\mu_F^2) =
        \int\limits_x^1\!\frac{dy}{y}\, \mathbf{P}(x/y,a_s) \cdot \mathbf{f}(y,\mu_F^2)

with :math:`\mathbf P` the Altarelli-Parisi splitting functions (see :doc:`pQCD`).
In :doc:`Mellin space <Mellin>` the |DGLAP| equations are just differential equations:

.. math::
    \frac{d}{d\ln(\mu_F^2)} \tilde{\mathbf{f}}(\mu_F^2) = -\gamma(a_s) \cdot \tilde{\mathbf{f}}(\mu_F^2)

(Note the additional minus in the definition for :math:`\gamma`).

We change the evolution variable to the (monotonic) :ref:`theory/pQCD:strong coupling` :math:`a_s(\mu_F^2)`
and the equations to solve become

.. math::
    \frac{d}{da_s} \tilde{\mathbf{f}}(a_s)
        = \frac{d\ln(\mu_F^2)}{da_s} \cdot \frac{d \tilde{\mathbf{f}}(\mu_F^2)}{d\ln(\mu_F^2)}
        = -\frac{\gamma(a_s)}{\beta(a_s)} \cdot \tilde{\mathbf{f}}(a_s)

This assumes the factorization scale :math:`\mu_F^2` (the inherit scale of the |PDF|) and the
renormalization scale :math:`\mu_R^2` (the inherit scale for the strong coupling) to be equal.

The (formal) solution can then be written in terms of an |EKO| :math:`\mathbf E` :cite:`Bonvini:2012sh`

.. math::
    \tilde{\mathbf{f}}(a_s) &= \tilde{\mathbf{E}}(a_s \leftarrow a_s^0) \cdot \tilde{\mathbf{f}}(a_s^0)\\
    \tilde{\mathbf{E}}(a_s \leftarrow a_s^0) &= \mathcal P \exp\left[-\int\limits_{a_s^0}^{a_s} \frac{\gamma(a_s')}{\beta(a_s')} da_s' \right]

with :math:`\mathcal P` the path-ordering operator. In the non-singlet sector the equations decouple and
we do not need to worry about neither matrices nor the path-ordering.

Using :doc:`Interpolation <Interpolation>` on both the initial and final |PDF|, we can then discretize the
|EKO| in x-space and define :math:`{\mathbf{E}}_{k,j}` (represented by
:class:`~eko.evolution_operator.Operator`) by

.. math::
    {\mathbf{E}}_{k,j}(a_s \leftarrow a_s^0) = \mathcal{M}^{-1}\left[\tilde{\mathbf{E}}(a_s \leftarrow a_s^0)\tilde p_j\right](x_k)

Now, we can write the solution to |DGLAP| in a true matrix operator scheme and find

.. math::
    \mathbf{f}(x_k,a_s) = {\mathbf{E}}_{k,j}(a_s \leftarrow a_s^0) \mathbf{f}(x_j,a_s^0)

so the |EKO| is a rank-4 operator acting both in flavor and momentum fraction space.

The issue of matching conditions when crossing flavor thresholds is discussed in a separate :doc:`document <Matching>`

LO evolution
------------

Expanding the anomalous dimension :math:`\gamma(a_s)` and the beta function :math:`\beta(a_s)`
to |LO| we obtain the (exact) |EKO|:

.. math::
    \ln \tilde {\mathbf E}^{(0)}(a_s \leftarrow a_s^0) &= \gamma^{(0)}\int\limits_{a_s^0}^{a_s} \frac{da_s'}{\beta_0 a_s'} = \gamma^{(0)} \cdot j^{(0,0)}(a_s,a_s^0)\\
    j^{(0,0)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s} \frac{da_s'}{\beta_0 a_s'} = \frac{\ln(a_s/a_s^0)}{\beta_0}

In |LO| we always use the *exact* solution.

Non-Singlet
^^^^^^^^^^^

We find

.. math::
    \frac{d}{da_s} \tilde f_{ns}^{(0)}(a_s) = \frac{\gamma_{ns}^{(0)}}{\beta_0 a_s}  \cdot \tilde f_{ns}^{(0)}(a_s)

with :math:`\gamma_{ns}^{(0)} = \gamma_{ns,+}^{(0)} = \gamma_{ns,-}^{(0)} = \gamma_{ns,v}^{(0)} = \gamma_{qq}^{(0)}`.

The |EKO| is then given by a simple exponential :cite:`Vogt:2004ns`

.. math::
    \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) = \exp\left[\gamma_{ns}^{(0)} \ln(a_s/a_s^0)/\beta_0 \right]

Singlet
^^^^^^^

We find

.. math::
    \frac{d}{da_s} \dSV{0}{a_s} = \frac{\gamma_S^{(0)}}{\beta_0 a_s} \cdot \dSV{0}{a_s}\,, \qquad
    \gamma_S^{(0)} = \begin{pmatrix}
                                \gamma_{qq}^{(0)} & \gamma_{qg}^{(0)}\\
                                \gamma_{gq}^{(0)} & \gamma_{gg}^{(0)}
                            \end{pmatrix}

In order to exponentiate the EKO, we decompose it
:math:`\ln \mathbf{\tilde E}^{(0)}_S = \lambda_+ {\mathbf e}_+ + \lambda_- {\mathbf e}_-` with
the eigenvalues :math:`\lambda_{\pm}` and the projectors :math:`\mathbf e_{\pm}` given by :cite:`Vogt:2004ns`

.. math::
    \lambda_{\pm} &= \frac 1 {2} \left( \ln \tilde E_{qq}^{(0)} + \ln \tilde E_{gg}^{(0)} \pm \sqrt{(\ln \tilde E_{qq}^{(0)}-\ln \tilde E_{gg}^{(0)})^2 + 4\ln \tilde E_{qg}^{(0)}\ln \tilde E_{gq}^{(0)}} \right)\\
    {\mathbf e}_{\pm} &= \frac{1}{\lambda_{\pm} - \lambda_{\mp}} \left( \ln \mathbf{\tilde E}^{(0)}_S - \lambda_{\mp} \mathbf I \right)

with :math:`\mathbf I` the 2x2 identity matrix in flavor space and, e.g., :math:`\ln \tilde E_{qq}^{(0)} = \gamma_{qq}^{(0)}j^{(0,0)}(a_s,a_s^0)`.

The projectors obey the usual properties, i.e.

.. math::
    {\mathbf e}_{\pm} \cdot {\mathbf e}_{\pm} = {\mathbf e}_{\pm}\,,\quad {\mathbf e}_{\pm} \cdot {\mathbf e}_{\mp} = 0\,,\quad \ep + \em = \mathbf I

and thus the exponentiation becomes easier again.

The |EKO| is then given by

.. math::
    \ESk{0}{a_s}{a_s^0} = \ep \exp(\lambda_{+}) + \em \exp(\lambda_{-})

NLO evolution
-------------

Non-Singlet
^^^^^^^^^^^

We find

.. math::
    \frac{d}{da_s} \tilde f_{ns}^{(1)}(a_s) = \frac{\gamma_{ns}^{(0)} a_s + \gamma_{ns}^{(1)} a_s^2}{\beta_0 a_s^2 + \beta_1 a_s^3} \cdot \tilde f_{ns}^{(1)}(a_s)

with :math:`\gamma_{ns} \in \{\gamma_{ns,+},\gamma_{ns,-}=\gamma_{ns,v}\}`.

We obtain the (exact) |EKO| :cite:`RuizArriola:1998er,Vogt:2004ns,Bonvini:2012sh`:

.. math::
    \ln \tilde E^{(1)}_{ns}(a_s \leftarrow a_s^0) &= \gamma^{(0)} \cdot j^{(0,1)}(a_s,a_s^0) + \gamma^{(1)} \cdot j^{(1,1)}(a_s,a_s^0)\\
    j^{(1,1)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'^2}{\beta_0 a_s'^2 + \beta_1 a_s'^3} = \frac{1}{\beta_1}\ln\left(\frac{1+b_1 a_s}{1+b_1 a_s^0}\right)\\
    j^{(0,1)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'}{\beta_0 a_s'^2 + \beta_1 a_s'^3} = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,1)}(a_s,a_s^0)

Note that we recover the |LO| solution:

.. math::
    \ln \tilde E^{(1)}_{ns}(a_s \leftarrow a_s^0) = \ln \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) + j^{(1,1)}(a_s,a_s^0)(\gamma^{(1)} - b_1 \gamma^{(0)})

In |NLO| we provide different strategies to define the |EKO|:

- ``method in ['iterate-exact', 'decompose-exact', 'perturbative-exact']``: use the *exact* solution as defined above
- ``method in ['iterate-expanded', 'decompose-expanded', 'perturbative-expanded']``: use the *exact* |LO| solution and substitute:

    .. math ::
        j^{(1,1)}(a_s,a_s^0) \to j^{(1,1)}_{exp}(a_s,a_s^0) &= \frac 1 {\beta_0}(a_s - a_s^0) \\
        j^{(0,1)}(a_s,a_s^0) \to j^{(0,1)}_{exp}(a_s,a_s^0) &= j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,1)}_{exp}(a_s,a_s^0) \\

- ``method = 'ordered-truncated'``: expanding the *argument* of the exponential of the new term but keeping the order we obtain:

.. math::
    \tilde E^{(1)}_{ns}(a_s \leftarrow a_s^0) = \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) \frac{1 + a_s/\beta_0 (\gamma_{ns}^{(1)} - b_1 \gamma_{ns}^{(0)})}{1 + a_s^0/\beta_0 (\gamma_{ns}^{(1)} - b_1 \gamma_{ns}^{(0)})}

- ``method = 'truncated'``: expanding the *whole* exponential of the new term we obtain:

.. math::
    \tilde E^{(1)}_{ns}(a_s \leftarrow a_s^0) = \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) \left[1 + (a_s - a_s^0)/\beta_0 (\gamma_{ns}^{(1)} - b_1 \gamma_{ns}^{(0)}) \right]

Singlet
^^^^^^^

We find

.. math::
    \frac{d}{da_s} \dSV{1}{a_s} = \frac{\gamma_{S}^{(0)} a_s + \gamma_{S}^{(1)} a_s^2}{\beta_0 a_s^2 + \beta_1 a_s^3} \cdot \dSV{1}{a_s}

with :math:`\gamma_{S}^{(0)} \gamma_{S}^{(1)} \neq \gamma_{S}^{(1)} \gamma_{S}^{(0)}`.

Here the strategies are:

- for ``method in ['iterate-exact', 'iterate-expanded']`` we use a discretized path-ordering :cite:`Bonvini:2012sh`:

.. math::
    \ESk{1}{a_s}{a_s^0} = \prod\limits_{k=n}^{0} \ESk{1}{a_s^{k+1}}{a_s^{k}}\quad \text{with}\quad a_s^{n+1} = a_s

where the order of the product is such that later |EKO| are to the left and

.. math::
    \ESk{1}{a_s^{k+1}}{a_s^{k}} &= \exp\left(-\frac{\gamma(a_s^{k+1/2})}{\beta(a_s^{k+1/2})} \Delta a_s \right) \\
    a_s^{k+1/2} &= a_0 + \left(k+ \frac 1 2\right) \Delta a_s\\
    \Delta a_s &= \frac{a_s - a_s^0}{n + 1}

using the projector algebra from |LO| to exponentiate the single steps.

- for ``method in ['decompose-exact', 'decompose-expanded']``: use the exact or the approximate exact
  integrals from the non-singlet sector and then decompose :math:`\ln \tilde{\mathbf E}^{(1)}` -
  this will neglect the non-commutativity of the singlet matrices.

- for ``method in ['perturbative-exact', 'perturbative-expanded', 'ordered-truncated', 'truncated']``
  we seek for an perturbative solution around the (exact) leading order operator:

  We set :cite:`Vogt:2004ns`

    .. math::
        \frac{d}{da_s} \dSV{1}{a_s} = \frac{\mathbf R (a_s)}{a_s} \cdot \dSV{1}{a_s}\,, \quad
        \mathbf R (a_s) = \sum\limits_{k=0} a_s^k \mathbf R_{k}

  where in |NLO| we find

    .. math::
        \mathbf R_0 = \gamma_{S}^{(0)}/\beta_0\,,\quad
        \mathbf R_1 = \gamma_{S}^{(1)}/\beta_0 - b_1 \gamma_{S}^{(0)} /\beta_0

  and for the higher coefficients

    - ``method = 'perturbative-exact'``: :math:`\mathbf R_k = - b_1 \mathbf R_{k-1}\,\text{for}\,k>1`
    - ``method = 'perturbative-expanded'``: :math:`\mathbf R_k = 0\,\text{for}\,k>1`

  We make an ansatz for the solution

    .. math::
        \ESk{1}{a_s}{a_s^0} = \mathbf U (a_s) \ESk{0}{a_s}{a_s^0} {\mathbf U}^{-1} (a_s^0), \quad
        \mathbf U (a_s) = \mathbf I + \sum\limits_{k=1} a_s^k \mathbf U_k

  Inserting this ansatz into the differential equation and sorting by powers of :math:`a_s`, we
  obtain a recursive set of commutator relations for the evolution operator coefficients
  :math:`\mathbf U_k`:

    .. math::
        [\mathbf U_1, \mathbf R_0] &= \mathbf R_1 - \mathbf U_1\\
        [\mathbf U_k, \mathbf R_0] &= \mathbf R_k + \sum\limits_{j=1}^{k-1} \mathbf R_{k-j} \mathbf U_j - k \mathbf U_k = \mathbf{R}_k' - k \mathbf U_k\,,k>1

  Multiplying these equations with :math:`\mathbf e_{\pm}` from left and right and using the identity

    .. math::
        \mathbf U_k = \em \mathbf U_k \em + \em \mathbf U_k \ep + \ep \mathbf U_k \em + \ep \mathbf U_k \ep

  we obtain the :math:`\mathbf U_k`

    .. math::
        \mathbf U_k = \frac{ \em \mathbf{R}_k' \em + \ep \mathbf{R}_k' \ep } k + \frac{\ep \mathbf{R}_k' \em}{r_- - r_+ + k} + \frac{\em \mathbf{R}_k' \ep}{r_+ - r_- + k}

  with :math:`r_{\pm} =\frac 1 {2\beta_0} \left( \gamma_{qq}^{(0)} + \gamma_{gg}^{(0)} \pm \sqrt{(\gamma_{qq}^{(0)}-\gamma_{gg}^{(0)})^2 + 4\gamma_{qg}^{(0)}\gamma_{gq}^{(0)}} \right)`.

  So the strategies are

    - ``method in ['perturbative-exact', 'perturbative-expanded']``: approximate the full evolution
      operator :math:`\mathbf U(a_s)` with an expansion up to ``ev_op_max_order``
    - ``method in ['ordered-truncated', 'truncated']``: truncate the evolution operator :math:`\mathbf U(a_s)` and use

    .. math::
        \ESk{1}{a_s}{a_s^0} = \ESk{0}{a_s}{a_s^0} + a_s \mathbf U_1 \ESk{0}{a_s}{a_s^0} - a_s^0 \ESk{0}{a_s}{a_s^0} \mathbf U_1

NNLO evolution
--------------

Non-Singlet
^^^^^^^^^^^

We find

.. math::
    \frac{d}{da_s} \tilde f_{ns}^{(2)}(a_s) = \frac{\gamma_{ns}^{(0)} a_s + \gamma_{ns}^{(1)} a_s^2 + \gamma_{ns}^{(2)} a_s^3 }{\beta_0 a_s^2 + \beta_1 a_s^3 + \beta_2 a_s^4} \cdot \tilde f_{ns}^{(2)}(a_s)

with :math:`\gamma_{ns} \in \{\gamma_{ns,+},\gamma_{ns,-},\gamma_{ns,v}\}`.

We obtain the (exact) |EKO| :cite:`Vogt:2004ns,Cafarella_2008`:

.. math::
    \ln \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) &= \gamma_{ns}^{(0)} j^{(0,2)}(a_s,a_s^0) + \gamma_{ns}^{(1)} j^{(1,2)}(a_s,a_s^0) + \gamma_{ns}^{(2)} j^{(2,2)}(a_s,a_s^0)\\

with:

.. math::
    j^{(2,2)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'^3}{\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4} = \frac{1}{\beta_2}\ln\left(\frac{1 + a_s ( b_1 + b_2 a_s ) }{ 1 + a_s^0 ( b_1 + b_2 a_s^0 )}\right) - \frac{b_1}{ \beta_2 \Delta} \delta \\
    \delta &= \arctan \left( \frac{b_1 + 2 a_s b_2 }{ \Delta} \right) - \arctan \left( \frac{b_1 + 2 a_s^0 b_2 }{ \Delta} \right) \\
        &= \frac{i}{2} \left[ ln \left( \frac{ \Delta - i (b_1 + 2a_s b_2)}{ \Delta + i (b_1 + 2a_s b_2)}\right) - ln \left( \frac{ \Delta - i (b_1 + 2a_s^0 b_2)}{ \Delta + i (b_1 + 2a_s^0 b_2)}\right) \right] \\
        &= \arctan \left( \frac{\Delta ( a_s - a_s^0 )}{ 2 + b_1 (a_s + a_s^0) + 2 a_s a_s^0 b_2 } \right) \\
    \Delta &= \sqrt{4 b_2 - b_1^2 }

and:

.. math::
    j^{(1,2)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'^2}{\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4} = \frac{2}{\beta_0 \Delta} \delta \\
    j^{(0,2)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'}{\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4} = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,2)}(a_s,a_s^0) - b_2 j^{(2,2)}(a_s,a_s^0)

Note, plugging the numerical values of :math:`\beta_i` we find that the :math:`\Delta \in \mathbb{R}` if :math:`n_f < 6`.
However you can notice that :math:`\Delta` appears always with :math:`\delta` and the fraction :math:`\frac{\delta}{\Delta} \in \mathbb{R}, \forall n_f`.

We can recover the |LO| solution:

.. math::
    \ln \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) = \ln \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) + j^{(1,2)}(a_s,a_s^0)(\gamma^{(1)} - b_1 \gamma^{(0)}) + j^{(2,2)}(a_s,a_s^0)(\gamma^{(2)} - b_2 \gamma^{(0)})

And thus the |NLO| solution:

.. math::
    \ln \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) &= \ln \tilde E^{(1)}_{ns}(a_s \leftarrow a_s^0) + j^{(1,2)'}(a_s,a_s^0)(\gamma^{(1)} - b_1 \gamma^{(0)}) + j^{(2,2)}(a_s,a_s^0)(\gamma^{(2)} - b_2 \gamma^{(0)}) \\
    j^{(1,2)'}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{ \beta_2 a_s'^2}{( \beta_0 + \beta_1 a_s' + \beta_2 a_s'^2 ) (\beta_0 + \beta_1 a_s')}

In |NNLO| we provide different strategies to define the |EKO|:

- ``method in ['iterate-exact', 'decompose-exact', 'perturbative-exact']``: use the *exact* solution as defined above
- ``method in ['iterate-expanded', 'decompose-expanded', 'perturbative-expanded']``: use the *exact* |LO| solution and expand all functions :math:`j^{(n,m)}(a_s,a_s^0)` to the order :math:`\mathcal o(a_s^3)`. We find:

.. math::
    j^{(2,2)}(a_s,a_s^0) \approx j^{(2,2)}_{exp}(a_s,a_s^0) &= \frac{1}{2\beta_0} (a_s^2 - (a_s^0)^{2}) \\
    j^{(1,2)}(a_s,a_s^0) \approx j^{(1,2)}_{exp}(a_s,a_s^0) &= \frac{1}{\beta_0} [ (a_s - a_s^0) - \frac{b_1}{2} (a_s^2 - (a_s^0)^{2})] \\
    j^{(0,2)}(a_s,a_s^0) \approx j^{(0,2)}_{exp}(a_s,a_s^0) &= j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,2)}_{exp}(a_s,a_s^0) - b_2 j^{(2,2)}_{exp}(a_s,a_s^0) \\
    &= j^{(0,0)}(a_s,a_s^0) - \frac{1}{\beta_0} [ b_1 (a_s - a_s^0) + \frac{b_1^2-b_2}{2} (a_s^2 - (a_s^0)^{2}) ] \\

This method corresponds to ``IMODEV=2`` of :cite:`Vogt:2004ns`.

- ``method = 'ordered-truncated'``: for this method we follow the prescription from :cite:`Vogt:2004ns` and we get:

.. math::
    \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) = \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) \frac{ 1 + a_s U_1 + a_s^2 U_2 }{ 1 + a_s^0 U_1 + (a_s^0)^{2} U_2 }

with the unitary matrices defined consistently with the method ``perturbative`` adopted for NLO singlet evolution:

.. math::
    U_1 &= R_1 = \frac{1}{\beta_0}[ \gamma^{(1)} - b_1 \gamma^{(0)}] \\
    U_2 &= \frac{1}{2}[ R_1^2 + R_2 ] \\
    R_2 &= \gamma_{ns}^{(2)}/\beta_0 - b_1 R_1 - b_2 R_0 \\

This method corresponds to ``IMODEV=3`` of :cite:`Vogt:2004ns`.

- ``method = 'truncated'``: we expand the *whole* exponential and keeping terms within :math:`\mathcal o(a_s^3)`. This method is the fastest among the ones provided by our program. We obtain:

.. math::
    \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) = \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) \left [ 1 + U_1 (a_s - a_s^0) + a_s^2 U_2 - a_s a_s^0 U_1^2 + (a_s^0)^{2} ( U_1^2 - U_2 ) \right]



Singlet
^^^^^^^

For the singlet evolution we find:

.. math::
    \frac{d}{da_s} \dSV{2}{a_s} = \frac{\gamma_{S}^{(0)} a_s + \gamma_{S}^{(1)} a_s^2 + \gamma_{S}^{(2)} a_s^3}{\beta_0 a_s^2 + \beta_1 a_s^3 + \beta_2 a_s^4} \cdot \dSV{2}{a_s}

with :math:`\gamma_{S}^{(i)} \gamma_{S}^{(j)} \neq \gamma_{S}^{(j)} \gamma_{S}^{(i)}, \quad i,j=0,1,2`.

In analogy to |NLO| we define the following strategies :

- for ``method in ['iterate-exact', 'iterate-expanded']`` we use a discretized path-ordering :cite:`Bonvini:2012sh`:

.. math::
    \ESk{2}{a_s}{a_s^0} = \prod\limits_{k=n}^{0} \ESk{2}{a_s^{k+1}}{a_s^{k}} \quad \text{with} \quad a_s^{n+1} = a_s

All the procedure is identical to |NLO|, simply the beat function is now expanded until :math:`\mathcal o(a_s^4)`

- for ``method in ['decompose-exact', 'decompose-expanded']``: use the exact or the approximate exact
  integrals from the non-singlet sector and then decompose :math:`\ln \tilde{\mathbf E}^{(2)}` -
  this will neglect the non-commutativity of the singlet matrices.

- for ``method in ['perturbative-exact', 'perturbative-expanded', 'ordered-truncated', 'truncated']``
  we seek for an perturbative solution around the (exact) leading order operator. We set :cite:`Vogt:2004ns`

    .. math::
        \frac{d}{da_s} \dSV{2}{a_s} = \frac{\mathbf R (a_s)}{a_s} \cdot \dSV{2}{a_s}\,, \quad
        \mathbf R (a_s) = \sum\limits_{k=0} a_s^k \mathbf R_{k}

  Finding one additional term compared to |NLO|:

    .. math::
        \mathbf R_2 & = \gamma_{S}^{(2)}/\beta_0 - b_1 \mathbf R_1 - b_2 \mathbf R_0 \\
        & = \frac{1}{\beta_0} [ \gamma_{S}^{(2)} - b_1 \gamma_{S}^{(1)} - \gamma_{S}^{(0)} ( b_2 - b_1^2 ) ]

  and for the higher coefficients

    - ``method = 'perturbative-exact'``: :math:`\mathbf R_k = - b_1 \mathbf R_{k-1} - b_2 \mathbf R_{k-2} \,\text{for}\,k>2`
    - ``method = 'perturbative-expanded'``: :math:`\mathbf R_k = 0\,\text{for}\,k>2`

    The solution ansatz becomes:

    .. math::
        \ESk{2}{a_s}{a_s^0} = \mathbf U (a_s) \ESk{0}{a_s}{a_s^0} {\mathbf U}^{-1} (a_s^0), \quad
        \mathbf U (a_s) = \mathbf I + \sum\limits_{k=1} a_s^k \mathbf U_k

  with:

    .. math::
        [\mathbf U_2, \mathbf R_0] &= \mathbf R_2 + \mathbf R_1 \mathbf U_1 - 2 \mathbf U_2\\

  So the strategies are:

    - ``method in ['perturbative-exact', 'perturbative-expanded']``: approximate the full evolution
      operator :math:`\mathbf U(a_s)` with an expansion up to ``ev_op_max_order``
    - ``method in ['ordered-truncated', 'truncated']``: truncate the evolution operator :math:`\mathbf U(a_s)` and use

    .. math::
        \ESk{2}{a_s}{a_s^0} &= \ESk{0}{a_s}{a_s^0} + a_s \mathbf U_1 \ESk{0}{a_s}{a_s^0} - a_s^0 \ESk{0}{a_s}{a_s^0} \mathbf U_1 \\
        &\hspace{20pt} + a_s^2 \mathbf U_2 \ESk{0}{a_s}{a_s^0} \\
        &\hspace{20pt} + a_s a_s^0 \mathbf U_1 \ESk{0}{a_s}{a_s^0} \mathbf U_1 \\
        &\hspace{20pt}- (a_s^0)^{2} \ESk{0}{a_s}{a_s^0} ( \mathbf U_1^2 - \mathbf U_2 )


N3LO evolution
--------------

Non-Singlet
^^^^^^^^^^^

At |N3LO| the |DGLAP| expansion reads:

.. math::
    \frac{d}{da_s} \tilde f_{ns}^{(2)}(a_s) = \frac{\gamma_{ns}^{(0)} a_s + \gamma_{ns}^{(1)} a_s^2 + \gamma_{ns}^{(2)} a_s^3 + \gamma_{ns}^{(3)} a_s^4 }{\beta_0 a_s^2 + \beta_1 a_s^3 + \beta_2 a_s^4 + \beta_3 a_s^5 } \cdot \tilde f_{ns}^{(2)}(a_s)

with :math:`\gamma_{ns} \in \{\gamma_{ns,+},\gamma_{ns,-},\gamma_{ns,v}\}`.

We obtain the (exact) |EKO| in analogy of the previous orders and recovering the |LO| solution:

.. math::
    \ln \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) &= \gamma_{ns}^{(0)} j^{(0,3)}(a_s,a_s^0) + \gamma_{ns}^{(1)} j^{(1,3)}(a_s,a_s^0) + \gamma_{ns}^{(2)} j^{(2,3)}(a_s,a_s^0) + \gamma_{ns}^{(3)} j^{(3,3)}(a_s,a_s^0)\\

with:

.. math::
    j^{(3,3)}(a_s,a_s^0) &= \frac{1}{\beta_0} \sum_{r=r_1}^{r_3} \frac{ r^2 [\ln(a_s-r) - \ln(a_s^0 - r)]}{b_1 + 2 b_2 r + 3 b_3 r^2} \\
    j^{(2,3)}(a_s,a_s^0) &= \frac{1}{\beta_0} \sum_{r=r_1}^{r_3} \frac{r [\ln(a_s-r) - \ln(a_s^0 - r)]}{b_1 + 2 b_2 r + 3 b_3 r^2} \\
    j^{(1,3)}(a_s,a_s^0) &= \frac{1}{\beta_0} \sum_{r=r_1}^{r_3} \frac{\ln(a_s-r) - \ln(a_s^0 - r)}{b_1 + 2 b_2 r + 3 b_3 r^2} \\
    j^{(0,3)}(a_s,a_s^0) &= j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,3)}(a_s,a_s^0) - b_2 j^{(2,3)}(a_s,a_s^0)- b_3 j^{(3,3)}(a_s,a_s^0)

where the sum is carried on the complex roots of the beta function expansion:

.. math ::
    a_s \in \{r_1, r_2, r_3 \} | \quad 1 + b_1 a_s + b_2 a_s^2 + b_3 a_s^3 = 0

You can notice that in the denominator of the integrals appears always the derivative of this expansion.
We remark that even though the roots are complex the total integral is real.

Also in this case we provide a we provide different strategies to define the |EKO|:

- ``method in ['iterate-exact', 'decompose-exact', 'perturbative-exact']``: use the *exact* solution as defined above
- ``method in ['iterate-expanded', 'decompose-expanded', 'perturbative-expanded']``: use the *exact* |LO| solution and expand all functions :math:`j^{(n,m)}(a_s,a_s^0)` to the order :math:`\mathcal o(a_s^3)`. We find:

.. math::
    j^{(3,3)}(a_s,a_s^0) &\approx j^{(3,3)}_{exp}(a_s,a_s^0) = \frac{1}{3 \beta_0} (a_s^3 - (a_s^0)^3) \\
    j^{(2,3)}(a_s,a_s^0) &\approx j^{(2,3)}_{exp}(a_s,a_s^0) = \frac{1}{\beta_0} [ \frac{1}{2} (a_s^2 - (a_s^0)^2) - \frac{b_1}{3} (a_s^3 - (a_s^0)^3) ]\\
    j^{(1,3)}(a_s,a_s^0) &\approx j^{(1,3)}_{exp}(a_s,a_s^0) = \frac{1}{\beta_0} [ (a_s - a_s^0) - \frac{b_1}{2} (a_s^2 - (a_s^0)^2) + \frac{b_1^2-b_2}{3} (a_s^3 - (a_s^0)^3) ]\\
    j^{(0,2)}(a_s,a_s^0) &\approx j^{(0,3)}_{exp}(a_s,a_s^0) = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,3)}_{exp}(a_s,a_s^0) - b_2 j^{(2,3)}_{exp}(a_s,a_s^0)- b_3 j^{(3,3)}_{exp}(a_s,a_s^0) \\



- ``method = 'ordered-truncated'``: performing the expansion one order higher wrt to |NNLO| we get:

.. math::
    \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) = \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0) \frac{ 1 + a_s U_1 + a_s^2 U_2 + a_s^3 U_3 }{ 1 + a_s^0 U_1 + (a_s^0)^{2} U_2 + (a_s^0)^{3} U_3 }

with the new unitary matrices defined:

.. math::
    U_3 &= \frac{1}{3} [R_3 + R_2 U_1 + R_1 U_2] \\
    R_3 &= \gamma_{ns}^{(3)}/\beta_0 - b_1 R_2 - b_2 R_1 - b_3 R_0 \\


- ``method = 'truncated'``:

.. math::
    \tilde E^{(2)}_{ns}(a_s \leftarrow a_s^0) = \tilde E^{(0)}_{ns}(a_s \leftarrow a_s^0)  & \left[ \right. 1 \\
        & + U_1 (a_s - a_s^0) \\
        & + a_s^2 U_2 - a_s a_s^0 U_1^2 + (a_s^0)^{2} ( U_1^2 - U_2 ) \\
        & + a_s^3 U_3 - a_s^2 a_s^0 U_2 U_1 + a_s (a_s^0)^{2} U_1 ( U_1^2 - U_2 ) - (a_s^0)^{3} (U_1^3 - 2 U_1 U_2 + U_3) \left. \right] \\

Singlet
^^^^^^^

For the singlet evolution we find:

.. math::
    \frac{d}{da_s} \dSV{2}{a_s} = \frac{\gamma_{S}^{(0)} a_s + \gamma_{S}^{(1)} a_s^2 + \gamma_{S}^{(2)} a_s^3 + \gamma_{S}^{(3)} a_s^3}{\beta_0 a_s^2 + \beta_1 a_s^3 + \beta_2 a_s^4 + \beta_3 a_s^5} \cdot \dSV{2}{a_s}

with :math:`\gamma_{S}^{(i)} \gamma_{S}^{(j)} \neq \gamma_{S}^{(j)} \gamma_{S}^{(i)}, \quad i,j=0,1,2,3`.

In analogy to |NLO| we define the following strategies :

- for ``method in ['iterate-exact', 'iterate-expanded']``: the solution strategies is exactly the same
  as in |NLO| and |NNLO| simply the beat function is now expanded until :math:`\mathcal o(a_s^5)`

- for ``method in ['decompose-exact', 'decompose-expanded']``: use the exact or the approximate exact
  integrals from the non-singlet sector and then decompose :math:`\ln \tilde{\mathbf E}^{(3)}` -
  this will neglect the non-commutativity of the singlet matrices.

- for ``method in ['perturbative-exact', 'perturbative-expanded', 'ordered-truncated', 'truncated']``
  we seek for an perturbative solution around the (exact) leading order operator. Following the notation used for
  previous orders we have:


    .. math::
        \mathbf R_2 & = \gamma_{S}^{(3)}/\beta_0 - b_1 \mathbf R_2 - b_2 \mathbf R_1 - b_3 \mathbf R_0 \\

  and for the higher coefficients:

    - ``method = 'perturbative-exact'``: :math:`\mathbf R_k = - b_1 \mathbf R_{k-1} - b_2 \mathbf R_{k-2} - b_3 \mathbf R_{k-3} \,\text{for}\,k>3`
    - ``method = 'perturbative-expanded'``: :math:`\mathbf R_k = 0\,\text{for}\,k>3`

  The new unitary matrix entering in the evolution ansatz follows the commutation relation:

    .. math::
        [\mathbf U_3, \mathbf R_0] &= \mathbf R_3 + \mathbf R_2 \mathbf U_1 + \mathbf R_1 \mathbf U_2 - 3 \mathbf U_3\\

  So the strategies are:

    - ``method in ['perturbative-exact', 'perturbative-expanded']``: approximate the full evolution
      operator :math:`\mathbf U(a_s)` with an expansion up to ``ev_op_max_order``
    - ``method in ['ordered-truncated', 'truncated']``: truncate the evolution operator :math:`\mathbf U(a_s)` and use

    .. math::
        \ESk{3}{a_s}{a_s^0} = \mathbf E^{0} &+ a_s \mathbf U_1 \mathbf E^{0} - a_s^0 \mathbf E^{0} \mathbf U_1 \\
        & + a_s^2 \mathbf U_2 \mathbf E^{0} + a_s a_s^0 \mathbf U_1 \mathbf E^{0} \mathbf U_1 - (a_s^0)^{2} \mathbf E^{0} ( \mathbf U_1^2 - \mathbf U_2 ) \\
        & + a_s^3 \mathbf U_3 \mathbf E^{0} - a_s^2 a_s^0 \mathbf U_2 \mathbf E^{0} \mathbf U_1 + a_s (a_s^0)^2 \mathbf U_1 E^{0} (\mathbf U_1^2 - \mathbf U_2) \\
        & - (a_s^0)^3 \mathbf E^{0} (\mathbf U_1^3 - \mathbf U_1 \mathbf U_2 - \mathbf U_2 \mathbf U_1 + \mathbf U_3) \\

    .. math ::
        \mathbf E^{0} = \ESk{0}{a_s}{a_s^0}

Intrinsic evolution
-------------------

We also consider the evolution of intrinsic heavy |PDF|. Since these are massive partons they can not
split any collinear particles and thus they do not participate in the |DGLAP| evolution. Instead, their
evolution is simply an identity operation: e.g. for an intrinsic charm distribution we get for
:math:`m_c^2 > Q_1^2 > Q_0^2`:

.. math ::
    \tilde c(Q_1^2) &= \tilde c(Q_0^2)\\
    \tilde {\bar c}(Q_1^2) &= \tilde{\bar c}(Q_0^2)

After :doc:`crossing the mass threshold </theory/Matching>` (charm in this example) the |PDF| can not be considered intrinsic
any longer and hence, they have to be rejoined with their evolution basis elements and take then again
part in the ordinary collinear evolution.

Mixed |QCD| :math:`\otimes` |QED| evolution
-----------------------------------------

For the moment in this case only the `exact` evolution is implemented.

Singlet
^^^^^^^

The evolution is obtained in the same way of the pure |QCD| case, with the only difference that
now both :math:`\gamma` and :math:`\beta_{qcd}` contain the |QED| corrections.

In the case in which :math:`\alpha_{em}` is running, at every step of the iteration the corresponding value
of :math:`a_{em}(a_s)` is used.

Non singlet
^^^^^^^^^^^

For the non singlet, being it diagonal, the solution is straightforward.
When :math:`\alpha_{em}` is fixed, the terms proportional to it are just a constant in the splitting functions, and therefore
they can be integrated directly. For example at ``order=(1,1)`` we have

.. math::
    \tilde E^{(1,1)}_{ns}(a_s \leftarrow a_s^0) &= \exp \Bigl( -\int_{\log \mu_0^2}^{\log \mu^2}d\log\mu^2 \gamma_{ns}^{(1,0)} a_s(\log\mu^2) + \gamma_{ns}^{(1,1)} a_s(\log\mu^2) a_{em} + \gamma_{ns}^{(0,1)} a_em \Bigr) \\
    & = \exp \Bigl( \int_{a_s^0}^{a_s}da_s\frac{\gamma_{ns}^{(1,0)} a_s + \gamma_{ns}^{(1,1)} a_s a_{em} + \gamma_{ns}^{(0,1)} a_em}{a_s^2(\beta_0 + \beta_0^{mix} a_{em})}  -\int_{\log \mu_0^2}^{\log \mu^2}d\log\mu^2 \gamma_{ns}^{(0,1)} a_em\Bigr)

In the last expression, the first term can be integrated with the :math:`j^{(n,m)` functions, while the second term is trivial.

In the case of :math:`\alpha_{em}` running, the :math:`a_s` integration integral is divided in steps, such that in every step
:math:`\alpha_{em}` is considered constant. In this way the solution will be the product of the solutions of every integration step:

.. math::
    \tilde E^{(1,1)}_{ns}(a_s \leftarrow a_s^0) = \prod\limits_{k=n}^{0} E^{(1,1)}_{ns}(a_s^{k+1} \leftarrow a_s^k, a_{em}^k)
