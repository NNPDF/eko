"""Collection of QED valence EKOs."""
import numba as nb

from .singlet_qed import eko_iterate, eko_perturbative, eko_truncated


@nb.njit(cache=True)
def dispatcher(
    order,
    method,
    gamma_valence,
    as_list,
    a_half,
    nf,
    ev_op_iterations,
    ev_op_max_order,
):
    """
    Determine used kernel and call it.

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
    aem_list : numpy.ndarray
        electromagnetic coupling values
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
            gamma_valence, as_list, a_half, nf, order, ev_op_iterations, dim=2
        )
    if method == "perturbative-exact":
        return eko_perturbative(
            gamma_valence,
            as_list,
            a_half,
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
            as_list,
            a_half,
            nf,
            order,
            ev_op_iterations,
            ev_op_max_order,
            False,
            dim=2,
        )
    if method in ["truncated", "ordered-truncated"]:
        return eko_truncated(
            gamma_valence, as_list, a_half, nf, order, ev_op_iterations, dim=2
        )
    raise NotImplementedError("Selected method is not implemented")
