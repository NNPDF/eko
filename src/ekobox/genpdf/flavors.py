"""Collection of flavor tools."""

import copy

import numpy as np

from eko import basis_rotation as br


def pid_to_flavor(pids):
    """Create flavor representations from PIDs.

    Parameters
    ----------
    pids : list(int)
        active PIDs

    Returns
    -------
    numpy.ndarray
        list of reprentations for each PID
    """
    ps = []
    zeros = np.zeros(len(br.flavor_basis_pids))
    for pid in pids:
        p = zeros.copy()
        idx = br.flavor_basis_pids.index(pid)
        p[idx] = 1.0
        ps.append(p)
    return np.array(ps)


def evol_to_flavor(labels):
    """Create flavor representations from evolution members.

    Parameters
    ----------
    labels : list(str)
        active evolution distributions

    Returns
    -------
    numpy.ndarray
        list of reprentations for each distribution
    """
    ps = []
    for label in labels:
        idx = br.evol_basis.index(label)
        ps.append(br.rotate_flavor_to_evolution[idx].copy())
    return np.array(ps)


def project(blocks, reprs):
    """Project some combination of flavors defined by reprs from the blocks.

    Parameters
    ----------
    blocks : list(dict)
        PDF blocks
    reprs : list(int)
        active distributions in flavor representation

    Returns
    -------
    list(dict) :
        filtered blocks
    """
    new_blocks = copy.deepcopy(blocks)
    for block in new_blocks:
        current_pids = block["pids"]
        current_data = block["data"].T
        if len(current_data) == 0:
            continue
        # load all flavors
        flavor_data = [np.zeros_like(current_data[0]) for pid in br.flavor_basis_pids]
        for pid, pdf in zip(current_pids, current_data):
            idx = br.flavor_basis_pids.index(pid)
            flavor_data[idx] = pdf
        flavor_data = np.array(flavor_data)
        new_data = np.zeros_like(flavor_data)
        for elem in reprs:
            proj = elem[:, np.newaxis] * elem
            new_data += proj @ flavor_data / (elem @ elem)
        block["pids"] = br.flavor_basis_pids
        block["data"] = np.array(new_data).T
    return new_blocks


def is_evolution_labels(labels):
    """Check whether the labels are provided in evolution basis.

    Parameters
    ----------
    labels : list()
        list of labels

    Returns
    -------
    bool :
        is evolution basis
    """
    for label in labels:
        if not isinstance(label, str):
            return False
        if label not in br.evol_basis:
            return False
    return True


def is_pid_labels(labels):
    """Check whether the labels are provided in flavor basis.

    Parameters
    ----------
    labels : list()
        list of labels

    Returns
    -------
    bool :
        is flavor basis
    """
    try:
        labels = np.array(labels, dtype=np.int_)
    except (ValueError, TypeError):
        return False
    for label in labels:
        # label might still be a list (for general projection)
        if not isinstance(label, np.int_):
            return False
        if label not in br.flavor_basis_pids:
            return False
    return True
