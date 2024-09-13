r"""The write-up of the matching conditions is given in :doc:`Matching
Conditions </theory/Matching>`."""

import numpy as np

from .. import basis_rotation as br
from .. import constants


def pids_from_intrinsic_evol(label, nf, normalize):
    r"""Obtain the list of pids with their corresponding weight, that are
    contributing to ``evol``.

    The normalization of the weights is only needed for the output rotation:

    - if we want to build e.g. the singlet in the initial state we simply have to sum
      to obtain :math:`\Sigma = u + \bar u + d + \bar d + \ldots`
    - if we want to rotate back in the output we have to *normalize* the weights:
      e.g. in nf=3 :math:`u = \frac 1 6 \Sigma + \frac 1 6 V + \ldots`

    The normalization can only happen here since we're actively cutting out some
    flavor (according to ``nf``).

    Parameters
    ----------
    label : str
        evolution label
    nf : int
        maximum number of light flavors
    normalize : bool
        normalize output

    Returns
    -------
    list(float)
        list of weights
    """
    try:
        evol_idx = br.evol_basis.index(label)
        is_evol = True
    except ValueError:
        is_evol = False
    if is_evol:
        weights = br.rotate_flavor_to_evolution[evol_idx].copy()
        for j, pid in enumerate(br.flavor_basis_pids):
            if nf < abs(pid) <= 6:
                weights[j] = 0
    else:
        weights = rotate_pm_to_flavor(label)
    # normalize?
    if normalize:
        norm = weights @ weights
        weights = weights / norm
    return weights


def get_range(evol_labels, qed):
    """Determine the number of light and heavy flavors participating in the
    input and output.

    Here, we assume that the T distributions (e.g. T15) appears *always*
    before the corresponding V distribution (e.g. V15).

    Parameters
    ----------
    qed : bool
        activate qed

    Returns
    -------
    nf_in : int
        number of light flavors in the input
    nf_out : int
        number of light flavors in the output
    """
    nf_in = 3
    nf_out = 3

    def update(label, qed):
        nf = 3
        if label[0] == "T":
            if not qed:
                nf = round(np.sqrt(int(label[1:]) + 1))
            else:
                if label[1:] == "d3":
                    nf = 3
                elif label[1:] == "u3":
                    nf = 4
                elif label[1:] == "d8":
                    nf = 5
                elif label[1:] == "u8":
                    nf = 6
                else:
                    raise ValueError(f"{label[1:]} is not possible")
        return nf

    for op in evol_labels:
        nf_in = max(update(op.input, qed), nf_in)
        nf_out = max(update(op.target, qed), nf_out)

    return nf_in, nf_out


def rotate_pm_to_flavor(label):
    """Rotate from +- basis to flavor basis.

    Parameters
    ----------
    label : str
        label

    Returns
    -------
    list(float)
        list of weights
    """
    # g and ph are unaltered
    if label in ["g", "ph"]:
        return br.rotate_flavor_to_evolution[br.evol_basis.index(label)].copy()
    # no it has to be a quark with + or - appended
    if label[0] not in br.quark_names or label[1] not in ["+", "-"]:
        raise ValueError(f"Invalid pm label: {label}")
    w = np.zeros(len(br.flavor_basis_pids))
    idx = br.flavor_basis_names.index(label[0])
    pid = br.flavor_basis_pids[idx]
    w[idx] = 1
    # + is +, - is -
    if label[1] == "+":
        w[br.flavor_basis_pids.index(-pid)] = 1
    else:
        w[br.flavor_basis_pids.index(-pid)] = -1
    return w


def rotate_matching(nf, qed, inverse=False):
    """Rotation between matching basis (with e.g. S,g,...V8 and c+,c-) and new
    true evolution basis (with S,g,...V8,T15,V15).

    Parameters
    ----------
    nf : int
        number of active flavors in the higher patch: to activate T15, nf=4
    qed : bool
        use QED?
    inverse : bool
        use inverse conditions?

    Returns
    -------
    dict
        mapping in dot notation between the bases
    """
    # the gluon and the photon do not care about new quarks
    m = {"g.g": 1.0, "ph.ph": 1.0}
    # already active distributions
    q = br.quark_names[nf - 1]
    if not qed:
        for k in range(2, nf):  # nf is the upper, so excluded
            n = k**2 - 1
            m[f"V{n}.V{n}"] = 1.0
            m[f"T{n}.T{n}"] = 1.0
        # the new contributions
        n = nf**2 - 1  # nf is pointing upwards
        for tot, oth, qpm in (("S", f"T{n}", f"{q}+"), ("V", f"V{n}", f"{q}-")):
            if inverse:
                m[f"{tot}.{tot}"] = (nf - 1.0) / nf
                m[f"{tot}.{oth}"] = 1.0 / nf
                m[f"{qpm}.{tot}"] = 1.0 / nf
                m[f"{qpm}.{oth}"] = -1.0 / nf
            else:
                m[f"{tot}.{tot}"] = 1.0
                m[f"{tot}.{qpm}"] = 1.0
                m[f"{oth}.{tot}"] = 1.0
                m[f"{oth}.{qpm}"] = -(nf - 1.0)
    else:
        names = {3: "d3", 4: "u3", 5: "d8", 6: "u8"}
        for k in range(3, nf):
            m[f"V{names[k]}.V{names[k]}"] = 1.0
            m[f"T{names[k]}.T{names[k]}"] = 1.0
        for tot, totdelta, oth, qpm in (
            ("S", "Sdelta", f"T{names[nf]}", f"{q}+"),
            ("V", "Vdelta", f"V{names[nf]}", f"{q}-"),
        ):
            a, b, c, d, e, f = qed_rotation_parameters(nf)
            if inverse:
                den = -b * d + a * e - c * e + b * f
                m[f"{tot}.{tot}"] = -(c * e - b * f) / den
                m[f"{tot}.{totdelta}"] = e / den
                m[f"{tot}.{oth}"] = -b / den
                m[f"{totdelta}.{tot}"] = (c * d - a * f) / den
                m[f"{totdelta}.{totdelta}"] = (f - d) / den
                m[f"{totdelta}.{oth}"] = (a - c) / den
                m[f"{qpm}.{tot}"] = (-b * d + a * e) / den
                m[f"{qpm}.{totdelta}"] = -e / den
                m[f"{qpm}.{oth}"] = b / den
            else:
                m[f"{tot}.{tot}"] = 1.0
                m[f"{tot}.{qpm}"] = 1.0
                m[f"{totdelta}.{tot}"] = a
                m[f"{totdelta}.{totdelta}"] = b
                m[f"{totdelta}.{qpm}"] = c
                m[f"{oth}.{tot}"] = d
                m[f"{oth}.{totdelta}"] = e
                m[f"{oth}.{qpm}"] = f
    # also higher quarks do not care
    for k in range(nf + 1, 6 + 1):
        q = br.quark_names[k - 1]
        for sgn in "+-":
            m[f"{q}{sgn}.{q}{sgn}"] = 1.0
    return m


def rotate_matching_inverse(nf, qed):
    """Inverse rotation between matching basis (with e.g. S,g,...V8 and c+,c-)
    and new true evolution basis (with S,g,...V8,T15,V15).

    Parameters
    ----------
    nf : int
        number of active flavors in the higher patch: to activate T15, nf=4
    qed : bool
        use QED?

    Returns
    -------
    dict
        mapping in dot notation between the bases
    """
    return rotate_matching(nf, qed, True)


def qed_rotation_parameters(nf):
    r"""Parameters of the QED basis rotation.

    From :math:`(\Sigma, \Sigma_{\Delta}, h_+)` into :math:`(\Sigma, \Sigma_{\Delta}, T_i^j)`,
    or equivalentely for :math:`V, V_{\Delta}, V_i^j, h_-`.

    Parameters
    ----------
    nf : int
        number of active flavors in the higher patch: e.g. to activate :math:`T_3^u` or :math:`V_3^u` choose ``nf=4``

    Returns
    -------
    a,b,c,d,e,f : float
        Parameters of the rotation: :math:`\Sigma_{\Delta} = a*\Sigma + b*\Sigma_{\Delta} + c*h_+, T_i = d*\Sigma + e*\Sigma_{\Delta} + f*h_+`
    """
    nu_l = constants.uplike_flavors(nf - 1)
    nd_l = (nf - 1) - nu_l
    nu_h = constants.uplike_flavors(nf)
    nd_h = nf - nu_h
    a = (nd_h / nu_h * nu_l - nd_l) / (nf - 1)
    b = nf / nu_h * nu_l / (nf - 1)
    c, d, e, f = (np.nan,) * 4
    if nf in [4, 6]:  # heavy flavor is up-like
        c = nd_h / nu_h
        d = nu_l / (nf - 1)
        e = nu_l / (nf - 1)
    elif nf in [3, 5]:  # heavy flavor is down-like
        c = -1
        d = nd_l / (nf - 1)
        e = -nu_l / (nf - 1)
    if nf in [3, 4]:  # s and c unlock T3d, T3u that have -h+
        f = -1
    elif nf in [5, 6]:  # b and t unlock T8d, T8u that have -2h+
        f = -2
    return a, b, c, d, e, f


def pids_from_intrinsic_unified_evol(label, nf, normalize):
    r"""Obtain the list of pids with their corresponding weight, that are
    contributing to intrinsic unified evolution.

    Parameters
    ----------
    evol : str
        evolution label
    nf : int
        maximum number of light flavors
    normalize : bool
        normalize output

    Returns
    -------
    list(float)
        list of weights
    """
    if label in ["ph", "g", "S", "V"]:
        return pids_from_intrinsic_evol(label, nf, normalize)
    if label[0] in br.quark_names[3:]:
        weights = rotate_pm_to_flavor(label)
    else:
        weights = np.array([0.0] * len(br.flavor_basis_pids))
        mapping = {
            "d3": {1: 1.0, 3: -1.0},  # T3d = d+ - s+
            "d8": {1: 1.0, 3: 1.0, 5: -2.0},  # T8d = d+ + s+ - 2b+
            "u3": {2: 1.0, 4: -1.0},  # T3u = u+ - c+
            "u8": {2: 1.0, 4: 1.0, 6: -2.0},  # T8u = u+ + c+ - 2t+
            "delta": {
                3: {2: 2.0, 1: -1.0, 3: -1.0},  # Sdelta = 2u+ - d+ -s+
                4: {2: 1.0, 4: 1.0, 1: -1.0, 3: -1.0},  # Sdelta = u+ + c+ - d+ - s+
                5: {
                    2: 3.0 / 2.0,
                    4: 3.0 / 2.0,
                    1: -1.0,
                    3: -1.0,
                    5: -1.0,
                },  # Sdelta = 3/2u+ + 3/2c+ - d+ - s+ - b+
                6: {
                    2: 1.0,
                    4: 1.0,
                    6: 1.0,
                    1: -1.0,
                    3: -1.0,
                    5: -1.0,
                },  # Sdelta = u+ + c+ + t+ - d+ -s+ - b+
            },
        }
        cur_map = mapping[label[1:]]
        if label[1:] == "delta":
            cur_map = cur_map[nf]
        for q, w in cur_map.items():
            weights[br.flavor_basis_pids.index(q)] = w
            weights[br.flavor_basis_pids.index(-q)] = (
                -1 if label[0] == "V" else 1.0
            ) * w

    # normalize?
    if normalize:
        norm = weights @ weights
        weights = weights / norm
    return weights
