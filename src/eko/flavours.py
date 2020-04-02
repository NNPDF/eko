# -*- coding: utf-8 -*-
r"""
    This module holds the class :class:`FlavourTarget` and the data
    related to the flavour combinations.


    Evolution Basis
    ---------------
    The evolution flavours are

    .. math ::
        V, V_3, V_8, V_{15}, V_{24}, V_{35}, T_3, T_8, T_{15}, T_{24}, T_{35}, \Sigma, g

    and they can be seperated into two major classes: the singlet sector, consisting of the
    singlet distribution :math:`\Sigma` and the gluon distribution :math:`g`, and the non-singlet
    sector. The non-singlet sector can be again subdivided into three groups: first the full
    valence distribution :math:`V`, second the valence-like distributions
    :math:`V_3 \ldots V_{35}`, and third the singlet like distributions :math:`T_3 \ldots T_{35}`.
    The mapping between the Evolution Basis and the Flavour Basis is given by

    .. math ::
        \Sigma &= \sum\limits_{j} q_j^+\\
        V &= \sum\limits_{j} q_j^-\\
        V_3 &= u^- - d^-\\
        V_8 &= u^- + d^- - 2 s^-\\
        V_{15} &= u^- + d^- + s^- - 3 c^-\\
        V_{24} &= u^- + d^- + s^- + c^- - 4 b^-\\
        V_{35} &= u^- + d^- + s^- + c^- + b^- - 5 t^-\\
        T_3 &= u^+ - d^+\\
        T_8 &= u^+ + d^+ - 2 s^+\\
        T_{15} &= u^+ + d^+ + s^+ - 3 c^+\\
        T_{24} &= u^+ + d^+ + s^+ + c^+ - 4 b^+\\
        T_{35} &= u^+ + d^+ + s^+ + c^+ + b^+ - 5 t^+

    with

    .. math ::
        q^\pm = q \pm \bar q

    Note that the associated numbers to the valence-like and singlet-like non-singlet distributions
    :math:`j` follow the common group-theoretical notation :math:`j = n_f^2 - 1`
    where :math:`n_f` denotes the incorporated number of quark flavours.

""" #pylint:disable=line-too-long

MINIMAL_NF = 3  # TODO Should this be a parameter or can it be hardcoded?

##### This can go to a separate file but for now it is ok here
# These are basically the parameters of the FlavourTarget class
# When nothing is given it is assumed nf_min = 3 and original == name
NSV = "NS_v"
NSP = "NS_p"
NSM = "NS_m"
VFNS = {
    "V": (4 * [NSV],),
    "V3": (4 * [NSM],),
    "V8": (4 * [NSM],),
    "V15": (3 * [NSM] + [NSV], "V", 4),
    "V24": (2 * [NSM] + 2 * [NSV], "V", 5),
    "V35": ([NSM] + 3 * [NSV], "V", 6),
    "T3": (4 * [NSP],),
    "T8": (4 * [NSP],),
    "T15": (3 * [NSP], ["S", "g"], 4),
    "T24": (2 * [NSP], ["S", "g"], 5),
    "T35": ([NSP], ["S", "g"], 6),
    "S": ([], ["S", "g"]),
    "g": ([], ["S", "g"]),
}

# TODO there are still a few hard-coded things in this class
# it would be nice to make it even more general
class FlavourTarget:
    """
        Defines the scheme

        Parameters
        ----------
            name: str
                name of the flavour target (T8, V8, etc)
            path: list(str)
                path to get to the target from the origin at the minimum possible nf
            original: str or list(str)
                original flavour name (or names if the result is a combination)
            nf_min: int
                minimal nf for which this flavour is active
    """

    def __init__(self, name, path, original=None, nf_min=None):
        self.name = name
        self._path = path
        self.nf_0 = MINIMAL_NF
        if nf_min is None:
            nf_min = self.nf_0
        if original is None:
            original = name
        self.nf_min = nf_min
        # In the most general case, the original input can be a combination
        self.base_flavour = original
        # And, actually, gluon and singlet are two special cases
        # TODO this is just a hack because I'm not able to generalize this part
        if name in ["S", "g"]:
            self.force_combination = True
        else:
            self.force_combination = False

    def _path_from_nf(self, nf_target, n_thres):
        """
            Generate the path array from the known nf target
            taking into account nf target == self.nf_0 would be
            the last member of the array

            Parameters
            ----------
                nf_target : int
                    number of flavours at target scale
                n_thres : int
                    numbers of thresholds to cross

            Returns
            -------
                path : list
                    path to get there
        """
        max_nns = len(self._path) + self.nf_0 - 1
        idx_ini = max(max_nns - nf_target, 0)
        idx_fin = idx_ini + n_thres + 1
        return self._path[idx_ini:idx_fin]

    def get_path(self, nf_target, n_thresholds):
        """
            Get the path to a given value of nf
            given a number of thresholds to be crossed

            Parameters
            ----------
                nf_target : int
                    nf value of the target flavour
                n_thresholds : int
                    number of thresholds which are going to be crossed

            Returns
            -------
                instructions : dict
                    a dictonary whose keys are the incoming flavour
                    and whose items are the corresponding path
        """
        # TODO can this be made more concise, or is this really the best that can be done?
        # Check from what nf we are coming from
        nf_from = nf_target - n_thresholds
        if nf_from < self.nf_0:
            raise ValueError(
                f"Physical configurations with less than {self.nf_0} were not considered"
            )
        # Now check from which flavour are we coming from
        if nf_from < self.nf_min or self.force_combination:
            flavour_from = self.base_flavour
        else:
            flavour_from = self.name
        # And now check whether this is a single flavour or a combination
        if isinstance(flavour_from, str):
            # Good, trivial
            return_path = [self._path_from_nf(nf_target, n_thresholds)]
            instructions = {flavour_from: return_path}
        else:  # oh, no...
            # Compute the depth of the singlet part of the path
            max_depth = n_thresholds + 1
            if self.force_combination:
                depth = max_depth
            else:
                depth = min(max_depth, self.nf_min - nf_from)
            # Find out the NS part of the path
            shift_th = n_thresholds - depth
            shift_nf = nf_target - depth
            ns_path = self._path_from_nf(shift_nf, shift_th)
            if self.name == "g":
                target_f = "g"
            else:
                target_f = "q"
            instructions = {}
            for flav in flavour_from:
                if flav == "S":
                    from_f = "q"
                elif flav == "g":
                    from_f = "g"
                paths = get_singlet_paths(target_f, from_f, depth)
                # Now insert in the paths the 'trivial' part
                for p in paths:
                    for extra in ns_path:
                        p.insert(0, extra)
                instructions[flav] = paths
        return instructions

def get_singlet_paths(to, fromm, depth):
    """
        Compute all possible path in the singlet sector to reach `to` starting from  `fromm`.

        Parameters
        ----------
            to : 'q' or 'g'
                final point
            fromm : 'q' or 'g'
                starting point
            depth : int
                nesting level; 1 corresponds to the trivial first step

        Returns
        -------
            ls : list(list(str))
                list of all possible paths, where each path is in increasing order, e.g.
                [P1(c <- a), P2(c <- a), ...] and P1(c <- a) = [(c <- b), (b <- a)]
    """
    if depth < 1:
        raise ValueError(f"Invalid arguments: depth >= 1, but got {depth}")
    if to not in ["q", "g"]:
        raise ValueError(f"Invalid arguments: to in [q,g], but got {to}")
    if fromm not in ["q", "g"]:
        raise ValueError(f"Invalid arguments: fromm in [q,g], but got {fromm}")
    # trivial?
    if depth == 1:
        return [[f"S_{to}{fromm}"]]
    # do recursion (if necessary, we could switch back to loops instead)
    qs = get_singlet_paths(to, "q", depth - 1)
    for q in qs:
        q.append(f"S_q{fromm}")
    gs = get_singlet_paths(to, "g", depth - 1)
    for g in gs:
        g.append(f"S_g{fromm}")
    return qs + gs

def get_all_flavour_paths(nf):
    """
        Builds all :class:`FlavourTarget`.

        Parameters
        ----------
            nf : None | int
                if given, paths are filtered by beeing active for at least `nf` flavours

        Returns
        -------
            ls : list(FlavourTarget)
                all instructions
    """
    ls = []
    # build flavour targets
    for flavour, data in VFNS.items():
        # Get the path
        path = data[0]
        base_flavour = None
        min_nf = None
        # Get the base_flavour if it is not itself
        if len(data) > 1:
            base_flavour = data[1]
        # Check at what nf this flavour enters for the first time
        if len(data) > 2:
            min_nf = data[2]
            if nf is not None and min_nf > nf:
                # Skip impossible values
                continue
        flt = FlavourTarget(flavour, path, original=base_flavour, nf_min=min_nf)
        ls.append(flt)
    return ls
