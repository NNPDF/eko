"""
    This module holds the classes that define the FNS

    Inside this class q is always treated as a q^2
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FlavourTarger:
    """
        Defines the scheme

        Parameters
        ----------
            `name`: str
                name of the flavour target (T8, V8, etc)
            `path`: list(str)
                path to get to the target from the origin at the minimum possible nf
            `original`: str
                original flavour name name
            `nf_min`: int
                minimal nf for which this flavour is active
            `protected`: bool
                whether flavours beyond the given nf can be obtained
    """
    def __init__(self, name, path, original, nf_min, protected = False):
        self.name = name
        self.path = path
        self.original = original
        self.nf_min = nf_min
        self.protected = protected

    def get_path(self, nf_target, thresholds):
        """ Get the path to a given value of nf
        given a number of thresholds to be crossed
        """
        # First check whether this flavour can be obtained
        if self.protected and nf_target < self.nf_min:
            return None
        # Check what the original flavour is for this case
        original_nf = nf_target - thresholds



##### Threshold scheme definition
# syntax:
# 
vfns = {
        'V' : ['NS_v', 'NS_v', 'NS_v', 'NS_v'],

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

        if scheme == "FFNS":
            if nf is None:
                logger.warning(
                    "No value for nf in the FFNS was received, defaulting to 5"
                )
                nf = 5
            if threshold_list is not None:
                raise ValueError("The FFNS does not accept any thresholds")
            self._areas = [Area(0, np.inf, self.q0, nf)]
            self.max_nf = nf
            self.min_nf = nf
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
