//! Library for anomalous dimension in |DGLAP| and |OME|.
//!
//! # Introduction
//!
//! `ekore` is a library for evolving collinear distributions in perturbative |QCD| and part of the [EKO framework](https://eko.readthedocs.io>).
//! In particular, it provides a common access to anomalous dimensions $\gamma$ and |OME| $\mathbf A$ needed for the evolution of |PDF| and similar objects.
//! The targeted objects obey, first, |DGLAP| like differential equations,
//! $$ \mu_F^2 \frac{d}{d\mu_F^2} \tilde{\mathbf{f}}(\mu_F^2) = -\gamma(a_s) \cdot \tilde{\mathbf{f}}(\mu_F^2) $$
//! and, second, a matching procedure, when changing the number of active flavors,
//! $$ \tilde{\mathbf{f}}^{(n_f+1)}(\mu^2)= {\mathbf{R}^{(n_f+1)}} \tilde{\mathbf{A}}^{(n_f+1)}(\mu^2, m^2) \tilde{\mathbf{f}}^{(n_f)}(\mu^2) \,, $$
//! where $ {\mathbf{R}^{(n_f+1)}} $ performs the necessary flavor basis rotation.
//!
//! # Available functions
//!
//! We currently support:
//! - unpolarized PDF: [anomalous dimensions][crate::anomalous_dimensions::unpolarized::spacelike], [operator matrix elements][crate::operator_matrix_elements::unpolarized::spacelike]
//! - longitudinally polarized PDF: [anomalous dimensions][crate::anomalous_dimensions::polarized::spacelike]
//!
//! Please refer to the individual functions to see the currently implemented perturbative accuracies.
//!
//! Note that we only cite the work from which we have implemented the object and we refer to this publication for further relevant citations and/or the original source.

// Let's stick to the original names which often come from FORTRAN, where such convention do not exists
#![allow(non_snake_case)]

pub mod anomalous_dimensions;
pub mod bib;
pub mod constants;
pub mod harmonics;
pub mod operator_matrix_elements;
mod util;
