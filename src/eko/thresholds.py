"""
    This module holds the classes that define the FNS
"""
import numpy as np

class EvolutionParams:
    """ Holds evolution parameters """
    def __init__(self, qini, qfin, nf):
        self.qini = qini
        self.qfin = qfin
        self.nf = nf

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

    def get_to(self, area_target):
        """ Returns the evolution necessary in order
        to get from this area to the target area.
        Target area should always be adjacent, but we deal with this later"""
        if area_target > self:
            go_to = area_target.qmin
        else:
            go_to = area_target.qmax

        if go_to == self.qref:
            return None
        else:
            return (self.qref, go_to)

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
        return target_area.qmin >= self.qmax

    def __lt__(self, target_area):
        return target_area.qmax <= self.qmin
    
    def __call__(self, q):
        return self.qmin <= q <= self.qmax

class Threshold:
    """ The threshold class holds information about the thresholds any
    Q has to pass in order to get there from a given q0 and scheme.

    Parameters
    ----------
        `setup`: dict
            Setup dictionary
        `scheme`: str
            Scheme definition
    """
    def __init__(self, setup, scheme = None):
        if scheme is None:
            scheme = setup.get("FNS", 'FFNS')
        self.q0 = setup["Q0"]
        self.areas = []
        self.bins = []
        self.area_q0 = 0
        if scheme == 'FFNS':
            nf = setup["NfFF"]
            self.areas = [Area(0, np.inf, self.q0, nf)]
        elif scheme == 'VFNS' or scheme == 'ZM-VFNS':
            self._setup_zm_vfns(setup)
        else:
            raise NotImplementedError(f"The scheme {scheme} not implemented in eko.dglap.py")

    def _setup_zm_vfns(self, setup):
        """ Receives the setup dictionary and sets up the zm_vfns scheme """
        Qmc = pow(setup.get("Qmc", 0),2)
        Qmb = pow(setup.get("Qmb", 0),2)
        Qmt = pow(setup.get("Qmt", 0),2)
        if Qmc > Qmb or Qmb > Qmt:
            raise ValueError("Quark masses are not in c < b < t order!")
        # Generate areas
        self.bins = [Qmc, Qmb, Qmt]
        self.areas = []
        nf = 3
        qmin = 0
        for i, qmax in enumerate(self.bins + [np.inf]):
            new_area = Area(qmin, qmax, self.q0, nf)
            if new_area.has_q0:
                self.area_q0 = i
            self.areas.append( new_area )
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
        if current_area < self.area_q0:
            rc = -1
        else:
            rc = 1
        area_path = [self.areas[i] for i in range(self.area_q0, current_area +rc ,rc)]
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
        if isinstance(qarr, (float,int)):
            qarr = np.array([qarr])
        # Check in which area is every q
        areas_idx = np.digitize(qarr, self.bins)
        return areas_idx

    def get_areas_q(self, qarr):
        """
        Returns the initial q for the area in which each value of qarr
        falls

        Parameters
        ----------
            `qarr`: np.array
                array of values of q

        Returns
        -------
            `areas_q`: list
                list of the reference area for each q
        """
        areas_idx = self.get_areas_idx(qarr)
        areas_q = [self.areas[i] for i in areas_idx]
        return areas_q


