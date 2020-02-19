"""
    This module holds the classes that define the FNS

    Inside this class q is always treated as a q^2
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
            Choice of scheme (default FFNS)
        `threshold_list`: list
            List of q^2 thresholds should the scheme accept it
        `nf`: int
            Number of flavour for the FFNS (default 5)
    """

    def __init__(self, qref=None, scheme="FFNS", threshold_list=None, nf=None):
        if qref is None:
            raise ValueError(
                "The threshold class needs to know about the reference q^{2}"
            )
        # Initial values
        self.q0 = qref
        self._areas = []
        self._area_walls = []
        self._area_ref = 0

        if scheme == "FFNS":
            if nf is None:
                logger.warning(
                    "No value for nf in the FFNS was received, defaulting to 5"
                )
                nf = 5
            if threshold_list is not None:
                raise ValueError("The FFNS does not accept any thresholds")
            self._areas = [Area(0, np.inf, self.q0, nf)]
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
        for i, qmax in enumerate(self._area_walls + [np.inf]):
            new_area = Area(qmin, qmax, qref, nf)
            if new_area.has_q0:
                self._area_ref = i
            self._areas.append(new_area)
            nf += 1
            qmin = qmax

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
        if isinstance(qarr, (float, int)):
            qarr = np.array([qarr])
        # Check in which area is every q
        areas_idx = np.digitize(qarr, self._area_walls)
        return areas_idx
