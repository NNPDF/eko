# -*- coding: utf-8 -*-
"""The following are the unpolarized time-like leading order Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np
from eko import constants



@nb.njit(cache=True)
def gamma_qq(N, s1):
	"""
	Computes the LO quark-quark anomalous dimension
	Implements Eqn. (B.3) from hep-ph/0604160

	Input parameters
	----------------
	N : Mellin moment (type: complex)
	s1 : harmonic sum $S_{1}$ (type: complex)

	Returns
	-------
	gamma_qq : LO quark-quark anomalous dimension $\gamma_{qq}^{(0)}(N)$ (type: complex)
	"""
	result = constants.CF * (-3.0 + (4.0 * s1) - 2.0 / (N * (N + 1.0)))
	return result

@nb.njit(cache=True)
def gamma_qg(N):
	"""
	Computes the LO quark-gluon anomalous dimension
	Implements Eqn. (B.4) from hep-ph/0604160 and Eqn. (A1) from PhysRevD.48.116

	Input parameters
	----------------
	N : Mellin moment (type: complex)

	Returns
	-------
	gamma_qg : LO quark-gluon anomalous dimension $\gamma_{qg}^{(0)}(N)$ (type: complex)
	"""
	result = - (N**2 + N + 2.0) / (N * (N + 1.0) * (N + 2.0))
	return result

@nb.njit(cache=True)
def gamma_gq(N, nf):
	"""
	Computes the LO gluon-quark anomalous dimension
	Implements Eqn. (B.5) from hep-ph/0604160 and Eqn. (A1) from PhysRevD.48.116

	Input parameters
	----------------
	N : Mellin moment (type: complex)
	nf : No. of active flavors (type: int)

	Returns
	-------
	gamma_qg : LO quark-gluon anomalous dimension $\gamma_{gq}^{(0)}(N)$ (type: complex)
	"""
	result = -4.0 * nf * constants.CF * (N**2 + N + 2.0) / (N * (N - 1.0) * (N + 1.0))
	return result

@nb.njit(cache=True)
def gamma_gg(N, s1, nf):
	"""
	Computes the LO gluon-gluon anomalous dimension
	Implements Eqn. (B.6) from hep-ph/0604160

	Input parameters
	----------------
	N : Mellin moment (type: complex)
	s1 : harmonic sum $S_{1}$ (type: complex)
	nf : No. of active flavors (type: int)

	Returns
	-------
	gamma_qq : LO quark-quark anomalous dimension $\gamma_{gg}^{(0)}(N)$ (type: complex)
	"""
	result = (2.0 * nf - 11.0 * constants.CA) / 3.0 + 4.0 * constants.CA * (s1 - 1.0 / (N * (N - 1.0)) - 1.0 / ((N + 1.0) * (N + 2.0)))
	return result

@nb.njit(cache=True)
def gamma_ns(N, s1):
	"""
	Computes the LO non-singlet anomalous dimension
	At LO, $\gamma_{ns}^{(0)} = \gamma_{qq}^{(0)}$
	
	Input parameters
	----------------
	N : Mellin moment (type: complex)
	s1 : harmonic sum $S_{1}$ (type: complex)

	Returns
	-------
	gamma_ns : LO quark-quark anomalous dimension $\gamma_{ns}^{(0)}(N)$ (type: complex)
	"""
	result = gamma_qq(N, s1)
	return result

@nb.njit(cache=True)
def gamma_singlet(N, s1, nf):
	"""
	Computes the LO singlet anomalous dimension matrix
	Implements Eqn. (2.13) from PhysRevD.48.116

	Input Parameters
	----------------
	N : Mellin moment (type: complex)
	s1 : harmonic sum $S_{1}$ (type: complex)
	nf : No. of active flavors (type: int)

	Returns
	-------
	gamma_singlet : LO singlet anomalous dimension matrix $\gamma_{s}^{(0)}$ (type: numpy.array)
	"""
	result = np.array([[gamma_qq(N, s1), gamma_gq(N, nf)],[gamma_qg(N), gamma_gg]], np.complex_)
	return result