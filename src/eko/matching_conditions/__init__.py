# -*- coding: utf-8 -*-
"""
This module defines the non-trivial matching conditions for the |VFNS| evolution.
"""
import numpy as np
from scipy import integrate
import numba as nb

from .. import mellin
from .. import interpolation
from ..operator.member import OpMember

from .nnlo import A_singlet_2, A_ns_2, A_hq_2, A_hg_2
from ..anomalous_dimensions import harmonics
from ..operator.flavors import singlet_labels, MemberName


# TODO: order might be removed if N3LO matching conditions will not be implemented

# TODO: sx can be computed inside quad ker to avaoid the repetition


@nb.njit("c16(u1,c16,f8)", cache=True)
def ome_ns_fact(order, n, a_s):
    """
    Build the non singlet operator matrix element

    Parameters
    ----------
        order : int
            perturbation order
        n : complex
            mellin moment
        nf : int
            number of active flavors
        a_s : float
            strong coupling constant
    Returns
    -------
        ome : complex
            non singlet operator matrix element
    """
    ome_ns = 1.0
    if order >= 2:
        sx = np.full(1, harmonics.harmonic_S1(n))
        sx = np.append(sx, harmonics.harmonic_S2(n))
        sx = np.append(sx, harmonics.harmonic_S3(n))
        ome_ns += a_s ** 2 * A_ns_2(n, sx)
    return ome_ns


@nb.njit("c16[:,:](u1,c16,f8)", cache=True)
def ome_singlet_fact(order, n, a_s):
    """
    Build the singlet operator matrix element

    Parameters
    ----------
        order : int
            perturbation order
        n : complex
            mellin moment
        nf : int
            number of active flavors
        a_s : float
            strong coupling constant
    Returns
    -------
        ome : complex
            singlet operator matrix element
    """
    # TODO: ome = np.identity(2, dtype=complex) not working???
    ome = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]])
    if order >= 2:
        sx = np.full(1, harmonics.harmonic_S1(n))
        sx = np.append(sx, harmonics.harmonic_S2(n))
        sx = np.append(sx, harmonics.harmonic_S3(n))
        ome += a_s ** 2 * A_singlet_2(n, sx)
    return ome


# TODO: maybe it is better for move this computation the nnlo module??
# to be more consistent wih S[0,0]
@nb.njit("c16[:](u1,c16,u1,f8)", cache=True)
def ome_thr_fact(order, n, nf, a_s):
    """
    Build the threshold operator matrix element

    Parameters
    ----------
        order : int
            perturbation order
        n : complex
            mellin moment
        nf : int
            number of active flavors
        a_s : float
            strong coupling constant
    Returns
    -------
        ome : complex
            threshold operator matrix element
    """
    ome = np.array([1+0j ,0+0j])
    if order >= 2:
        sx = np.full(1, harmonics.harmonic_S1(n))
        sx = np.append(sx, harmonics.harmonic_S2(n))
        sx = np.append(sx, harmonics.harmonic_S3(n))
        ome += a_s ** 2 * np.array( [ A_ns_2(n, sx) - nf * A_hq_2(n, sx), - nf * A_hg_2(n, sx)])
    return ome

# TODO: move quad ker inside the class ??
@nb.njit("f8(f8,u1,u1,string,b1,f8,f8[:,:],f8)", cache=True)
def quad_ker(
    u,
    order,
    nf,
    mode,
    is_log,
    logx,
    areas,
    a_s
):
    """
    Raw kernel inside quad

    Parameters
    ----------
        u : float
            quad argument
        order : int
            perturbation order
        nf : int
            number of active flavors
        mode : str
            element in the singlet sector
        is_log : boolean
            logarithmic interpolation
        logx : float
            Mellin inversion point
        areas : tuple
            basis function configuration
        a_s : float
            strong coupling constant evaluated at the m_quark haeavy scale

    Returns
    -------
        ker : float
            evaluated integration kernel
    """
    is_singlet = mode[0] == ("S" or "g")
    is_threshold = mode[0] == "T"
    # get transformation to N integral
    if is_singlet:
        r, o = 0.4 * 16.0 / (1.0 - logx), 1.0
    # elif is_threshold:
    #     # TODO: check this
    #     r, o = 0.5, 0.0
    else:
        r, o = 0.5, 0.0

    n = mellin.Talbot_path(u, r, o)
    jac = mellin.Talbot_jac(u, r, o)
    # check PDF is active
    if is_log:
        pj = interpolation.log_evaluate_Nx(n, logx, areas)
    else:
        pj = interpolation.evaluate_Nx(n, logx, areas)

    if pj == 0.0:
        return 0.0
    # compute the actual evolution kernel
    if is_singlet:
        ker = ome_singlet_fact(order, n, a_s)
        # select element of matrix
        k = 0 if mode[-1] == "S" else 1
        l = 0 if mode[-1] == "g" else 1
        ker = ker[k, l]

    elif is_threshold:
        # get nf from mode:
        ker = ome_thr_fact(order, n, nf, a_s)
        # select element of matrix
        k = 0 if mode[-1] == "S" else 1
        ker = ker[k]

    else:
        ker = ome_ns_fact(order, n, a_s)

    # recombine everthing
    mellin_prefactor = complex(0.0, -1.0 / np.pi)
    return np.real(mellin_prefactor * ker * pj * jac)


class OperatorMatrixElement:
    """
    Internal representation of a single Operator Matrix Element.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
        config : dict
            configuration
        managers : dict
            managers
        nf : int
            number of active flavors
        m2_heavy : float
            squared heavy quark mass
        mellin_cut : float
            cut to the upper limit in the mellin inversion
    """

    def __init__(self, config, managers, nf, m2_heavy, mellin_cut=1e-2):
        self.order = config["order"]
        self.nf = nf

        # compute a_s
        sc = managers["strong_coupling"]
        fact_to_ren = config["fact_to_ren"]
        self.m2_heavy = m2_heavy
        self.a_s = sc.a_s( m2_heavy / fact_to_ren, m2_heavy)

        self.int_disp = managers["interpol_dispatcher"] 
        self._mellin_cut = mellin_cut
        self.ome_members = {}
        self.op_members = {}


    def labels(self):
        """
        Determaine necessary sector labels to compute.

        Returns
        -------
            labels : list(str)
                sector labels
        """
        labels = []
        # V and T have NS contribution
        for f in range(2, self.nf + 1):
            j = f ** 2 - 1
            labels.append(f"V{j}.V{j}")
            labels.append(f"T{j}.V{j}")
        # Threshold operator
        j = (self.nf + 1) ** 2 - 1
        labels.append(f"V{j}.V{j}")
        labels.append(f"T{j}.S")
        labels.append(f"T{j}.g")
        # V is NS and T get singlet contribution
        for f in range(self.nf + 2 , 7):
            j = f ** 2 - 1
            # TODO: check that matmul is working with this syntax
            labels.append(f"V{j}.V{j}")
            labels.append(f"T{j}S.S")
            labels.append(f"T{j}S.g")
            labels.append(f"T{j}g.S")
            labels.append(f"T{j}g.g")
        labels.extend(["S.S","S.g","g.g","g.S","V.V",])

        return labels

    def compute(self):
        """ compute the actual operators (i.e. run the integrations) """

        # init all ops with zeros
        grid_size = len(self.int_disp.xgrid)
        for n in self.labels():
            self.ome_members[n] = OpMember(
                np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
            )

        # iterate output grid
        for k, logx in enumerate(np.log(self.int_disp.xgrid_raw)):
            # iterate basis functions
            for l, bf in enumerate(self.int_disp):
                # iterate sectors
                temp_labels = list(["S.S","S.g","g.g","g.S","V.V",])
                thr = (self.nf + 1) ** 2 - 1
                temp_labels.extend([f"T{thr}.S", f"T{thr}.g"])
                for label in temp_labels:
                    # compute and set
                    res = integrate.quad(
                        quad_ker,
                        0.5,
                        1.0 - self._mellin_cut,
                        args=(
                            self.order,
                            self.nf,
                            label,
                            self.int_disp.log,
                            logx,
                            bf.areas_representation,
                            self.a_s
                        ),
                        epsabs=1e-12,
                        epsrel=1e-5,
                        limit=100,
                        full_output=1,
                    )
                    val, err = res[:2]
                    self.ome_members[label].value[k][l] = val
                    self.ome_members[label].error[k][l] = err

        # copy repeated operator matrix elements
        self.copy_omes()

        # map the correct names
        self.ad_to_evol_map()

    def copy(self, out_label, in_label):
        """
        Copy value and error of one single operator matrix element

        Parameters
        ----------
            out_label : str
                final label
            in_label : str
                OME's label to be copied
        """
        self.ome_members[f"{out_label}"].value = self.ome_members[f"{in_label}"].value.copy()
        self.ome_members[f"{out_label}"].error = self.ome_members[f"{in_label}"].error.copy()

    def copy_omes(self):
        """Copy non-singlet and singlet operator matrix elements"""
        # V and T have NS contribution
        for f in range(2, self.nf + 1):
            j = f ** 2 - 1
            self.copy(f"V{j}.V{j}", "V.V")
            self.copy(f"T{j}.V{j}", "V.V")

        # Threshold operator, copy only V
        j = (self.nf + 1) ** 2 - 1
        self.copy(f"V{j}.V{j}", "V.V")

        # V is NS and T gets singlet contribution
        for f in range(self.nf + 2 , 7):
            j = f ** 2 - 1
            self.copy(f"V{j}.V{j}", "V.V")
            for label in ["S.S","S.g","g.g","g.S"]:
                self.copy(f"T{j}{label}", f"{label}")

    # TODO: remove this
    def ad_to_evol_map(self):
        for k, v in self.ome_members.items():
            self.op_members[MemberName(k)] = v.copy()
