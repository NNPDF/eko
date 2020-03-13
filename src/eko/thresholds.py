"""
    This module holds the classes that define the FNS: Threshold,
    Area, FlavourTarget
"""
# TODO there are still a few hard-coded things in this class
# it would be nice to make it even more general
import logging
import numpy as np
from eko.utils import get_singlet_paths

logger = logging.getLogger(__name__)

MINIMAL_NF = 3  # TODO Should this be a parameter or can it be hardcoded?


class FlavourTarget:
    """
        Defines the scheme

        Parameters
        ----------
            `name`: str
                name of the flavour target (T8, V8, etc)
            `path`: list(str)
                path to get to the target from the origin at the minimum possible nf
            `original`: str or list(str)
                original flavour name (or names if the result is a combination)
            `nf_min`: int
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
                `nf_target`: int
                    nf value of the target flavour
                `threshold`: int
                    number of thresholds which are going to be crossed

            Returns
            -------
                `instructions`: dict
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


class Area:
    """
        Sets up a single threhold area

        Parameters
        ----------
            q2_min : float
                lower bound of the area
            q2_max : float
                upper bound of the area
            q2_0 : float
                reference point of the area (can be somewhere in the area)
            nf : float
                number of flavours in the area
    """

    def __init__(self, q2_min, q2_max, q2_0, nf):
        self.q2_min = q2_min
        self.q2_max = q2_max
        self.nf = nf
        self.has_q2_0 = False
        # Now check which is the q2_ref for this area
        if q2_0 > q2_max:
            self.q2_ref = q2_max
        elif q2_0 < q2_min:
            self.q2_ref = q2_min
        else:
            self.has_q2_0 = True
            self.q2_ref = q2_0

    def q2_towards(self, q2):
        """
            Return q2_min or q2_max depending on whether
            we are going towards the max or the min or q2
            if we are alreay in the correct area

            Parameters
            ----------
                q2 : float
                    reference point

            Returns
            -------
                q2_next : float
                    the closest point to q2 that is within the area
        """
        if q2 > self.q2_max:
            return self.q2_max
        elif q2 < self.q2_min:
            return self.q2_min
        else:
            return q2

    def __gt__(self, target_area):
        """Compares q2 of areas"""
        return self.q2_min >= target_area.q2_max

    def __lt__(self, target_area):
        """Compares q2 of areas"""
        return self.q2_max <= target_area.q2_min

    def __call__(self, q2):
        """
            Checks whether q2 is contained in the area

            Parameters
            ----------
                q2 : float
                    testing point

            Returns
            -------
                contained : bool
                    is point contained?
        """
        return self.q2_min <= q2 <= self.q2_max


class Threshold:
    """
        The threshold class holds information about the thresholds any
        Q2 has to pass in order to get there from a given q2_ref and scheme.

        Parameters
        ----------
            `q2_ref` : float
                Reference q^2
            `scheme`: str
                Choice of scheme
            `threshold_list`: list
                List of q^2 thresholds should the scheme accept it
            `nf`: int
                Number of flavour for the FFNS
    """

    def __init__(self, q2_ref=None, scheme=None, threshold_list=None, nf=None):
        if q2_ref is None:
            raise ValueError(
                "The threshold class needs to know about the reference q^2"
            )
        # Initial values
        self.q2_ref = q2_ref
        self._threshold_list = []
        self._areas = []
        self._area_walls = []
        self._area_ref = 0
        self.scheme = scheme
        self.max_nf = None
        self.min_nf = None
        self._operator_paths = []
        protection = False

        if scheme == "FFNS":
            if nf is None:
                raise ValueError("No value for nf in the FFNS was received")
            if threshold_list is not None:
                raise ValueError("The FFNS does not accept any thresholds")
            self._areas = [Area(0, np.inf, self.q2_ref, nf)]
            self.max_nf = nf
            self.min_nf = nf
            protection = True
        elif scheme in ["VFNS", "ZM-VFNS"]:
            if nf is not None:
                logger.warning(
                    "The VFNS configures its own value for nf, ignoring input nf=%d", nf
                )
            if threshold_list is None:
                raise ValueError(
                    "The VFNS scheme was selected but no thresholds were input"
                )
            self._setup_vfns(threshold_list)
        else:
            raise NotImplementedError(
                f"The scheme {scheme} not implemented in eko.dglap.py"
            )

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
                if protection and min_nf > nf:
                    # Skip impossible values
                    continue
            flt = FlavourTarget(flavour, path, original=base_flavour, nf_min=min_nf)
            self._operator_paths.append(flt)

    @property
    def nf_ref(self):
        return self._areas[self._area_ref].nf

    def nf_range(self):
        """
            Iterate number of flavours, including min_nf *and* max_nf

            Yields
            ------
                nf : int
                    number of flavour
        """
        return range(self.min_nf, self.max_nf + 1)

    def _setup_vfns(self, threshold_list):
        """
            Receives a list of thresholds and sets up the vfns scheme

            Parameters
            ----------
                `threshold_list`: list
                    List of q^2 thresholds
        """
        nf = 3
        # Force sorting
        self._area_walls = sorted(threshold_list)
        # Generate areas
        self._areas = []
        q2_min = 0
        q2_ref = self.q2_ref
        self.min_nf = nf
        for i, q2_max in enumerate(self._area_walls + [np.inf]):
            nf = self.min_nf + i
            new_area = Area(q2_min, q2_max, q2_ref, nf)
            if new_area.has_q2_0:
                self._area_ref = i
            self._areas.append(new_area)
            q2_min = q2_max
        self.max_nf = nf

    def get_path_from_q2_ref(self, q2):
        """
            Get the Area path from q2_ref to q2.

            Parameters
            ----------
                `q2`: float
                    Target value of q2

            Returns
            -------
                `area_path`: list
                    List of Areas to go through in order to get from q2_ref
                    to q2. The first one is the one containg q2_ref while the
                    last one contains q2
        """
        current_area = self.get_areas_idx(q2)[0]
        if current_area < self._area_ref:
            rc = -1
        else:
            rc = 1
        area_path = [
            self._areas[i] for i in range(self._area_ref, current_area + rc, rc)
        ]
        return area_path

    def get_composition_path(self, nf, n_thres):
        """
            Iterates all flavour targets.

            Parameters
            ----------
                `nf_target`: int
                    nf value of the target flavour
                `threshold`: int
                    number of thresholds which are going to be crossed

            Yields
            ------
                name : string
                    flavour name
                path : list
                    flavour path
        """
        for flavour in self._operator_paths:
            yield flavour.name, flavour.get_path(nf, n_thres)

    def get_areas_idx(self, q2arr):
        """
            Returns the index of the area in which each value of q2arr falls.

            Parameters
            ----------
                `q2arr`: np.array
                    array of values of q2

            Returns
            -------
                `areas_idx`: list
                    list with the indices of the corresponding areas for q2arr
        """
        # Ensure q2arr is an array
        if isinstance(q2arr, (np.float, np.int, np.integer)):
            q2arr = np.array([q2arr])
        # Check in which area is every q2
        areas_idx = np.digitize(q2arr, self._area_walls)
        return areas_idx

    def get_areas(self, q2arr):
        """
            Returns the Areas in which each value of q2arr falls

            Parameters
            ----------
                `q2arr`: np.array
                    array of values of q2

            Returns
            -------
                `areas`: list
                    list with the areas for q2arr
        """
        idx = self.get_areas_idx(q2arr)
        area_list = np.array(self._areas)[idx]
        return list(area_list)
