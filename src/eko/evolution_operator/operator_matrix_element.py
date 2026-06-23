"""The |OME| for the non-trivial matching conditions in the |VFNS|
evolution."""

import copy
import functools
import logging
from typing import Optional

import numpy as np

from .. import basis_rotation as br
from .. import scale_variations as sv
from ..io.types import InversionMethod
from ..matchings import Segment
from . import Managers, Operator
from .quad_ker import MatchingMethods
from .quad_ker import quad_ker_ome as quad_ker

logger = logging.getLogger(__name__)


def matching_method(s: Optional[InversionMethod]) -> MatchingMethods:
    """Return the matching method.

    Parameters
    ----------
    s :
        string representation

    Returns
    -------
    i :
        int representation
    """
    if s is not None:
        return MatchingMethods["BACKWARD_" + s.value.upper()]
    return MatchingMethods.FORWARD


class OperatorMatrixElement(Operator):
    r"""Internal representation of a single |OME|.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
    config : dict
        configuration
    managers : dict
        managers
    nf: int
        number of active flavor below threshold
    q2: float
        squared matching scale
    is_backward: bool
        True for backward matching
    L: float
        :math:`\ln(\mu_F^2 / m_h^2)`
    is_msbar: bool
        add the |MSbar| contribution
    """

    log_label = "Matching"
    # complete list of possible matching operators labels
    full_labels = (
        *br.singlet_labels,
        (br.matching_hplus_pid, 21),
        (br.matching_hplus_pid, 100),
        (21, br.matching_hplus_pid),
        (100, br.matching_hplus_pid),
        (br.matching_hplus_pid, br.matching_hplus_pid),
        (200, 200),
        (200, br.matching_hminus_pid),
        (br.matching_hminus_pid, 200),
        (br.matching_hminus_pid, br.matching_hminus_pid),
    )
    # still valid in QED since Sdelta and Vdelta matchings are diagonal
    full_labels_qed = copy.deepcopy(full_labels)

    def __init__(
        self,
        config,
        managers: Managers,
        nf: int,
        q2: float,
        is_backward: bool,
        L: float,
        is_msbar: bool,
    ):
        super().__init__(config, managers, Segment(q2, q2, nf))
        self.backward_method = matching_method(
            config["backward_inversion"] if is_backward else None
        )
        self.L = L
        self.is_msbar = is_msbar
        # Note for the moment only QCD matching is implemented
        self.order = tuple(config["matching_order"])

    @property
    def labels(self):
        """Necessary sector labels to compute.

        Returns
        -------
        list(str)
            sector labels
        """
        labels = []
        # non-singlet labels
        if self.config["debug_skip_non_singlet"]:
            logger.warning("%s: skipping non-singlet sector", self.log_label)
        else:
            labels.append((200, 200))
            # intrinsic labels, which are not zero at NLO
            labels.append((br.matching_hminus_pid, br.matching_hminus_pid))
            # These contributions are always 0 for the moment
            # labels.extend([(200, br.matching_hminus_pid), (br.matching_hminus_pid, 200)])
        # same for singlet
        if self.config["debug_skip_singlet"]:
            logger.warning("%s: skipping singlet sector", self.log_label)
        else:
            labels.extend(
                [
                    *br.singlet_labels,
                    (br.matching_hplus_pid, 21),
                    (br.matching_hplus_pid, 100),
                ]
            )
            labels.extend(
                [
                    (21, br.matching_hplus_pid),
                    (100, br.matching_hplus_pid),
                    (br.matching_hplus_pid, br.matching_hplus_pid),
                ]
            )
        return labels

    def quad_ker(self, label, logx, areas):
        """Return partially initialized integrand function.

        Parameters
        ----------
        label: tuple
            operator element pids
        logx: float
            Mellin inversion point
        areas : tuple
            basis function configuration

        Returns
        -------
        functools.partial
            partially initialized integration kernel
        """
        return functools.partial(
            quad_ker,
            order=self.order,
            mode0=label[0],
            mode1=label[1],
            is_log=self.int_disp.log,
            logx=logx,
            areas=areas,
            a_s=self.a_s,
            nf=self.nf,
            L=self.L,
            sv_mode=self.sv_mode,
            Lsv=np.log(self.xif2),
            backward_method=self.backward_method,
            is_msbar=self.is_msbar,
            is_polarized=self.config["polarized"],
            is_time_like=self.config["time_like"],
        )

    @property
    def a_s(self):
        """Return the computed values for :math:`a_s`.

        Note that here you need to use :math:`a_s^{n_f+1}`
        """
        sc = self.managers.couplings
        return sc.a_s(
            self.q2_from
            * (self.xif2 if self.sv_mode == sv.Modes.exponentiated else 1.0),
            nf_to=self.nf + 1,
        )

    def compute(self):
        """Compute the actual operators (i.e. run the integrations)."""
        self.initialize_op_members()

        # At LO you don't need anything else
        if self.order[0] == 0:
            logger.info("%s: no need to compute matching at LO", self.log_label)
            return
        logger.info(
            "%s: order: (%d, %d), backward method: %s",
            self.log_label,
            self.order[0],
            self.order[1],
            MatchingMethods(self.backward_method).name,
        )

        self.integrate()
