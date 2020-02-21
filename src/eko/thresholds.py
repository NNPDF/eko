"""
    This module holds the classes that define the FNS

    Inside this class q is always treated as a q^2
"""
# TODO there are still a few hard-coded things in this class
# it would be nice to make it even more general
import numpy as np
import logging
from eko.utils import get_singlet_paths

logger = logging.getLogger(__name__)

MINIMAL_NF = 3 # Should this be a parameter or can it be hardcoded?

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
    def __init__(self, name, path, original = None, nf_min = None):
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
        # TODO this is just a hack because I'm not clever enough to generalize this part
        if name in ['S', 'g']:
            self.force_combination = True
        else:
            self.force_combination = False

    def _path_from_nf(self, nf_target, n_thres):
        """ Generate the path array from the known nf target
        Taking into account nf target == self.nf_0 would be
        the last member of the array
        """
        max_nns = len(self._path) + self.nf_0 - 1
        idx_ini = max_nns - nf_target
        idx_fin = idx_ini + n_thres + 1
        return self._path[idx_ini:idx_fin]

    def get_path(self, nf_target, n_thresholds):
        """ Get the path to a given value of nf
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
        # Check from what nf we are coming from
        nf_from = nf_target - n_thresholds
        if nf_from < self.nf_0:
            raise ValueError(f"Physical configurations with less than {self.nf_0} were not considered")
        # Now check from which flavour are we coming from
        if nf_from < self.nf_min or self.force_combination:
            flavour_from = self.base_flavour
        else:
            flavour_from = self.name
        import ipdb
        ipdb.set_trace()
        # And now check whether this is a single flavour or a combination
        if isinstance(flavour_from, str):
            # Good, trivial
            return_path = [self._path_from_nf(nf_target, n_thresholds)]
            instructions = { flavour_from : return_path }
        else: # oh, no...
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
            if self.name == 'g':
                target_f = 'g'
            else:
                target_f = 'q'
            instructions = {}
            for flav in flavour_from:
                if flav == 'S':
                    from_f = 'q'
                elif flav == 'g':
                    from_f = 'g'
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
NSV = 'NS_v'
NSP = 'NS_p'
NSM = 'NS_m'
VFNS = {
        'V' : (4*[NSV],),
        'V3' : (4*[NSM],),
        'V8' : (4*[NSM],),
        'V15' : (3*[NSM] + [NSV], 'V', 4),
        'V24' : (2*[NSM] + 2*[NSV], 'V', 5),
        'V35' : ([NSM] + 3*[NSV], 'V', 6),
        'T3' : (4*[NSP],),
        'T8' : (4*[NSP],),
        'T15' : (3*[NSP], ['S', 'g'], 4),
        'T24' : (2*[NSP], ['S', 'g'], 5),
        'T35' : ([NSP], ['S', 'g'], 6),
        'S' : ([], ['S', 'g']),
        'g' : ([], ['S', 'g']),
        }

class Area:
    """ Sets up an area """

    def __init__(self, qmin, qmax, q0, nf):
        self.qmin = qmin
        self.qmax = qmax
        self.nf = nf
        self.has_q0 = False
        # Now check which is the qref for this area
        if q0 > qmax:
            self.qref = qmax
        elif q0 < qmin:
            self.qref = qmin
        else:
            self.has_q0 = True
            self.qref = q0

    def q_towards(self, q):
        """ Return qmin or qmax depending on whether
        we are going towards the max or the min or q
        if we are alreay in the correct area """
        if q > self.qmax:
            return self.qmax
        elif q < self.qmin:
            return self.qmin
        else:
            return q

    def __gt__(self, target_area):
        return self.qmin >= target_area.qmax

    def __lt__(self, target_area):
        return self.qmax <= target_area.qmin

    def __call__(self, q):
        """ Checks whether q is contained in the area """
        return self.qmin <= q <= self.qmax


class Threshold:
    """ The threshold class holds information about the thresholds any
    Q has to pass in order to get there from a given q0 and scheme.

    Parameters
    ----------
        `qref` : float
            Reference q^2
        `scheme`: str
            Choice of scheme 
        `threshold_list`: list
            List of q^2 thresholds should the scheme accept it
        `nf`: int
            Number of flavour for the FFNS (default 5)
    """

    def __init__(self, qref=None, scheme=None, threshold_list=None, nf=None):
        if qref is None:
            raise ValueError(
                "The threshold class needs to know about the reference q^{2}"
            )
        # Initial values
        self.q0 = qref
        self._threshold_list = []
        self._areas = []
        self._area_walls = []
        self._area_ref = 0
        self._scheme = scheme
        self.max_nf = None
        self.min_nf = None
        self._operator_paths = []
        protection = False

        if scheme == "FFNS":
            if nf is None:
                raise ValueError("No value for nf in the FFNS was received")
            if threshold_list is not None:
                raise ValueError("The FFNS does not accept any thresholds")
            self._areas = [Area(0, np.inf, self.q0, nf)]
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
    def qref(self):
        return self.q0

    @property
    def nf_ref(self):
        return self._areas[self._area_ref].nf

    def nf_range(self):
        return range(self.min_nf, self.max_nf+1)

    def _setup_vfns(self, threshold_list):
        """ Receives a list of thresholds and sets up the vfns scheme

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
        qmin = 0
        qref = self.q0
        self.min_nf = nf
        for i, qmax in enumerate(self._area_walls + [np.inf]):
            new_area = Area(qmin, qmax, qref, nf)
            if new_area.has_q0:
                self._area_ref = i
            self._areas.append(new_area)
            nf += 1
            qmin = qmax
        self.max_nf = nf


    def get_path_from_q0(self, q):
        """ Get the Area path from q0 to q.

        Parameters
        ----------
            `q`: float
                Target value of q

        Returns
        -------
            `area_path`: list
                List of Areas to go through in order to get from q0
                to q. The first one is the one containg q0 while the
                last one contains q
        """
        current_area = self.get_areas_idx(q)[0]
        if current_area < self._area_ref:
            rc = -1
        else:
            rc = 1
        area_path = [
            self._areas[i] for i in range(self._area_ref, current_area + rc, rc)
        ]
        return area_path

    def get_areas_idx(self, qarr):
        """
        Returns the initial q for the area in which each value of qarr
        falls

        Parameters
        ----------
            `qarr`: np.array
                array of values of q

        Returns
        -------
            `areas_idx`: list
                list with the indices of the corresponding areas for qarr
        """
        # Ensure qarr is an array
        if isinstance(qarr, (np.float, np.int, np.integer)):
            qarr = np.array([qarr])
        # Check in which area is every q
        areas_idx = np.digitize(qarr, self._area_walls)
        return areas_idx

    def get_areas(self, qarr):
        """ Returns the Areas in which each value of qarr falls

        Parameters
        ----------
            `qarr`: np.array
                array of values of q
        Returns
        -------
            `areas`: list
                list with the areas for qarr
        """
        idx = self.get_areas_idx(qarr)
        area_list = np.array(self._areas)[idx]
        return list(area_list)
