"""Main application class of eko."""
import copy
import logging
import os

import numpy as np

from .. import interpolation, msbar_masses
from ..couplings import Couplings, couplings_mod_ev
from ..evolution_operator.grid import OperatorGrid
from ..io import EKO, Operator, runcards
from ..io.types import QuarkMassSchemes
from ..thresholds import ThresholdsAtlas
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

    def __init__(self, theory_card: dict, operators_card: dict, path: os.PathLike):
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
        elif new_theory.quark_masses_scheme is QuarkMassSchemes.POLE:
            masses = tuple(mq**2 for mq in new_theory.quark_masses)
        else:
            raise ValueError(f"Unknown mass scheme '{new_theory.quark_masses_scheme}'")

        # call explicitly iter to explain the static analyzer that is an
        # iterable
        thresholds_ratios = list(iter(new_theory.matching))
        tc = ThresholdsAtlas(
            masses=masses,
            q2_ref=new_operator.mu20,
            nf_ref=new_theory.num_flavs_init,
            thresholds_ratios=thresholds_ratios,
            max_nf=new_theory.num_flavs_max_pdf,
        )

        # strong coupling
        sc = Couplings(
            couplings_ref=np.array(new_theory.couplings.values),
            scale_ref=new_theory.couplings.alphas.scale**2,
            thresholds_ratios=thresholds_ratios,
            masses=tuple(m2 / new_theory.fact_to_ren**2 for m2 in masses),
            order=new_theory.order,
            method=couplings_mod_ev(new_operator.configs.evolution_method.value),
            nf_ref=new_theory.num_flavs_ref,
            max_nf=new_theory.num_flavs_max_as,
        )
        # setup operator grid
        self.op_grid = OperatorGrid(
            mu2grid=new_operator.mu2grid,
            order=new_theory.order,
            masses=masses,
            mass_scheme=new_theory.quark_masses_scheme.value,
            intrinsic_flavors=new_theory.intrinsic_flavors,
            fact_to_ren=new_theory.fact_to_ren,
            configs=new_operator.configs,
            debug=new_operator.debug,
            thresholds_config=tc,
            strong_coupling=sc,
            interpol_dispatcher=bfd,
        )

        self.out = EKO.create(path).load_cards(new_theory, new_operator).build()

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
