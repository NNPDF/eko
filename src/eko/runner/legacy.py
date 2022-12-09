"""Main application class of eko."""
import copy
import logging

from .. import interpolation, msbar_masses
from ..couplings import Couplings
from ..evolution_operator.grid import OperatorGrid
from ..io import EKO, Operator, runcards
from ..thresholds import ThresholdsAtlas
from ..types import QuarkMassSchemes
from . import commons

logger = logging.getLogger(__name__)


class Runner:
    """Represents a single input configuration.

    For details about the configuration, see :doc:`here </code/IO>`

    Attributes
    ----------
    setup : dict
        input configurations

    """

    banner = commons.BANNER

    def __init__(self, theory_card: dict, operators_card: dict):
        """Initialize runner.

        Parameters
        ----------
        theory_card: dict
            theory parameters and options
        operators_card: dict
            operator specific options

        """
        new_theory, new_operator = runcards.update(theory_card, operators_card)

        # Store inputs
        self._theory = new_theory

        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher(
            xgrid=new_operator.rotations.xgrid,
            polynomial_degree=new_operator.configs.interpolation_polynomial_degree,
        )

        # setup the Threshold path, compute masses if necessary
        masses = None
        if new_theory.quark_masses_scheme is QuarkMassSchemes.MSBAR:
            masses = msbar_masses.compute(new_theory)

        tc = ThresholdsAtlas(
            masses=(mq.value**2 for mq in new_theory.quark_masses),
            q2_ref=new_theory.num_flavs_ref.scale,
            nf_ref=new_theory.num_flavs_ref.value,
            thresholds_ratios=list(new_theory.matching),
            max_nf=new_theory.num_flavs_max_pdf,
        )

        # strong coupling
        sc = Couplings.from_dict(new_theory, masses=masses)
        # setup operator grid
        self.op_grid = OperatorGrid.from_dict(new_theory, new_operator, tc, sc, bfd)

        self.out = EKO.new(theory=theory_card, operator=new_operator)

    def get_output(self) -> EKO:
        """Run evolution and generate output operator.

        Two steps are applied sequentially:

        1. evolution is performed, computing the evolution operator in an
           internal flavor and x-space basis
        2. bases manipulations specified in the runcard are applied

        Returns
        -------
        EKO
            output instance

        """
        # add all operators
        for final_scale, op in self.op_grid.compute().items():
            self.out[float(final_scale)] = Operator.from_dict(op)

        return copy.deepcopy(self.out)
