"""Collection of QED valence EKOs."""
import numba as nb
import numpy as np

from .singlet_qed import eko_iterate, eko_perturbative, eko_truncated


@nb.njit(cache=True)
def dispatcher(
    order,
    method,
    gamma_valence,
    a1,
    a0,
    aem_list,
    nf,
    ev_op_iterations,
    ev_op_max_order,
):
    """
    Determine used kernel and call it.

    In LO we always use the exact solution.

    Parameters
    ----------
        order : tuple(int,int)
            perturbative order
        method : str
            method
        gamma_singlet : numpy.ndarray
            singlet anomalous dimensions matrices
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        aem : float
            electromagnetic coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps
        ev_op_max_order : tuple(int,int)
            perturbative expansion order of U

    Returns
    -------
        e_v : numpy.ndarray
            singlet EKO
    """
    if method in ["iterate-exact", "iterate-expanded"]:
        return eko_iterate(
            gamma_valence, a1, a0, aem_list, nf, order, ev_op_iterations, dim=2
        )
    if method == "perturbative-exact":
        return eko_perturbative(
            gamma_valence,
            a1,
            a0,
            aem_list,
            nf,
            order,
            ev_op_iterations,
            ev_op_max_order,
            True,
            dim=2,
        )
    if method == "perturbative-expanded":
        return eko_perturbative(
            gamma_valence,
            a1,
            a0,
            aem_list,
            nf,
            order,
            ev_op_iterations,
            ev_op_max_order,
            False,
            dim=2,
        )
    if method in ["truncated", "ordered-truncated"]:
        return eko_truncated(
            gamma_valence, a1, a0, aem_list, nf, order, ev_op_iterations, dim=2
        )
    raise NotImplementedError("Selected method is not implemented")
