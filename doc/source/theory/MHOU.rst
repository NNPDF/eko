Estimating Missing Higher Order Uncertainties
=============================================

Both, the beta function :math:`\beta(a_s)` and the anomalous dimensions :math:`\gamma(a_s)`,
are perturbatively calculated object and their full expression is never fully known.
Thus, an exact all-order solution of the |DGLAP| equations can never be calculated and,
moreover, the solution is unique (see :doc:`here</theory/DGLAP>`).
So, it is interesting to give an estimate of the missing higher order uncertainties (|MHOU|) :cite:`AbdulKhalek:2019ihb`
to account for this imperfect knowledge.

In order to do this we can provide several, independent prescriptions of which we require

    * the difference with the central solution has to be beyond the known perturbative orders
    * respect |RGE| invariance
    * universality: the same procedure can be used for any perturbative process
    * only use the number of flavors available locally

In the following we assume a variation by a factor :math:`\rho`, which range can not be
determined a priori.

Shifting Anomalous Dimensions
-----------------------------

This feature is active if ``ModSV='exponentiated'``. It corresponds to Eq. (3.32) of :cite:`AbdulKhalek:2019ihb`
and the procedure in :cite:`Vogt:2004ns` (note that here the new anomalous dimension is expanded in :math:`a_s(\rho Q^2)`
and so the log is inverted to :cite:`Vogt:2004ns`).

We make an ansatz

.. math ::
    \gamma\left(a_s(Q^2)\right) &= \bar \gamma\left(a_s(\rho Q^2),\rho\right) + O\left(\left(\alpha_s(Q^2)\right)^{N+2}\right)\\
    \Rightarrow \sum_{j=0}^N \left(a_s(Q^2)\right)^{1+j} \gamma_j &= \sum_{j=0}^N \left(a_s(\rho Q^2)\right)^{1+j} \bar \gamma_j(\rho)  + O\left(\left(\alpha_s(Q^2)\right)^{N+2}\right)

which we can solve, by defining the scale varied anomalous dimensions :math:`\bar\gamma_j` with

.. math ::
    \bar \gamma_0(\rho) &= \gamma_0\\
    \bar \gamma_1(\rho) &= \gamma_1 + \gamma_0 \beta_0\ln\rho \\
    \bar \gamma_2(\rho) &= \gamma_2 + \gamma_0 \beta_0^2\ln^2\rho + \left(2\gamma_1 \beta_0 + \gamma_0 \beta_1\right)\ln\rho \\
    \bar \gamma_3(\rho) &= \gamma_3 + \gamma_0 \beta_0^3\ln^3\rho + \left(3\gamma_1 \beta_0^2 + \frac 5 2 \gamma_0 \beta_0 \beta_1\right)\ln^2\rho\\
        &\hspace{20pt} + \left(3\gamma_2 \beta_0 + 2\gamma_1 \beta_1 + \gamma_0 \beta_2\right)\ln\rho

This procedure is repeated for each flavor patch present in the evolution path.

Shifting PDF Matching Conditions Matrix
---------------------------------------

This feature is active if ``ModSV='exponentiated'``. It is not mentiond in :cite:`AbdulKhalek:2019ihb`
as this paper deals only with a |FFNS| scenario and slightly implicit in :cite:`Vogt:2004ns`
as this paper only deals with :math:`a_s^2(\mu^2)` contributions to the matching conditions,
due to the lack of shifted matching points and intrinsic contributions.

We make an ansatz

.. math ::
    \mathbf{A}\left(a_s(m_h^2),\lambda_f\right) &= \bar{\mathbf{A}}\left(a_s(\rho m_h^2),\lambda_f,\rho\right) + O\left(\left(\alpha_s(m_h^2)\right)^{N+1}\right)\\
    \Rightarrow \sum_{j=0}^N \left(a_s(m_h^2)\right)^{j} \mathbf{A}_{j}(\lambda_f) &= \sum_{j=0}^N \left(a_s(\rho m_h^2)\right)^{j} \bar{\mathbf{A}}_{j}(\lambda_f,\rho)  + O\left(\left(\alpha_s(m_h^2)\right)^{N+1}\right)

which can be solve analogously to the case of anomalous dimensions above.

This procedure is repeated for each matching present in the evolution path.

Shifting Strong Coupling Matching Point
---------------------------------------

This feature is active if ``ModSV='exponentiated'``. It is not mentiond in :cite:`AbdulKhalek:2019ihb`,
but corresponds to the procedure in :cite:`Vogt:2004ns`.

We can shift the matching point of the strong coupling :math:`a_s(m_h^2)` to estimate the
|MHOU| related to the respective matching decoupling parameters. For ``ModSV='exponentiated'``
we match at :math:`a_s(\rho m_h^2)`, which naturally ensures the consistency of used
number of flavors.


Adding an EKO
-------------

This feature is active if ``ModSV='expanded'`` and it corresponds to Eq. (3.35) of :cite:`AbdulKhalek:2019ihb`.

We make an ansatz

.. math ::
    \bar{\mathbf{f}}(Q^2,\rho) = \mathbf{K}(a_s(\rho Q^2),\rho) \mathbf{E}(\rho Q^2 \leftarrow Q^2) \mathbf{f}(Q^2)

with

.. math ::
    \mathbf{1} &= \mathbf{K}(a_s(\rho Q^2),\rho) \mathbf{E}(\rho Q^2 \leftarrow Q^2) + O\left(\left(\alpha_s(Q^2)\right)^{N+2}\right) \\
      &= \sum_{j=0}^{N+1} \left(a_s(\rho Q^2)\right)^{j} \mathbf{K}_j(\rho) \mathbf{E}(\rho Q^2 \leftarrow Q^2) + O\left(\left(\alpha_s(Q^2)\right)^{N+2}\right)

which we can solve by defining the scale variation kernel :math:`K_j` with

.. math ::
    \mathbf{K}_0(\rho) &= \mathbf{1}\\
    \mathbf{K}_1(\rho) &= \gamma_0 \log\rho\\
    \mathbf{K}_2(\rho) &= \frac 1 2 \gamma_0 \left(\beta_0 + \gamma_0\right) \log^2\rho + \gamma_1 \log\rho\\
    \mathbf{K}_3(\rho) &= \frac 1 6 \gamma_0 \left(2\beta_0^2 + 3\beta_0\gamma_0 + \gamma_0^2\right) \log^3\rho\\
        &\hspace{20pt} + \frac 1 2 \left(\beta_1\gamma_0 + 2\beta_0\gamma_1 + \gamma_0\gamma_1 + \gamma_1\gamma_0\right) \log^2\rho + \gamma_2 \log\rho

This procedure is applied only to the last flavor patch present in the evolution path.

Note, that it is also possible and common to reattribute :math:`\mathbf{K}` instead to the hard matrix element.
