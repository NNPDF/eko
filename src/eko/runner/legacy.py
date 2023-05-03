"""Main application class of eko."""
import logging
import os
from typing import Union

import numpy as np

from .. import interpolation
from ..couplings import Couplings, couplings_mod_ev
from ..evolution_operator.grid import OperatorGrid
from ..io import EKO, Operator, runcards
from ..io.types import RawCard, ScaleVariationsMethod
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

    def __init__(
        self,
        theory_card: Union[RawCard, runcards.TheoryCard],
        operators_card: Union[RawCard, runcards.OperatorCard],
        path: os.PathLike,
    ):
        """Initialize runner.

        Parameters
        ----------
        theory_card :
            theory parameters and options
        operators_card :
            operator specific options
        path :
            path where to store the computed operator

        """
        new_theory, new_operator = runcards.update(theory_card, operators_card)

        # Store inputs
        self.path = path
        self._theory = new_theory

        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher(
            xgrid=new_operator.xgrid,
            polynomial_degree=new_operator.configs.interpolation_polynomial_degree,
        )

        # setup the Threshold path, compute masses if necessary
        masses = runcards.masses(new_theory, new_operator.configs.evolution_method)

        # call explicitly iter to explain the static analyzer that is an
        # iterable
        thresholds_ratios = np.power(list(iter(new_theory.heavy.matching_ratios)), 2.0)
        tc = ThresholdsAtlas(
            masses=masses,
            q2_ref=new_operator.mu20,
            nf_ref=new_theory.heavy.num_flavs_init,
            thresholds_ratios=thresholds_ratios,
            max_nf=new_theory.heavy.num_flavs_max_pdf,
        )

        # strong coupling
        sc = Couplings(
            couplings=new_theory.couplings,
            order=new_theory.order,
            method=couplings_mod_ev(new_operator.configs.evolution_method),
            masses=masses,
            hqm_scheme=new_theory.heavy.masses_scheme,
            thresholds_ratios=thresholds_ratios
            * (
                new_theory.xif**2
                if new_operator.configs.scvar_method
                == ScaleVariationsMethod.EXPONENTIATED
                else 1.0
            ),
        )
        # setup operator grid
        self.op_grid = OperatorGrid(
            mu2grid=new_operator.mu2grid,
            order=new_theory.order,
            masses=masses,
            mass_scheme=new_theory.heavy.masses_scheme.value,
            intrinsic_flavors=new_theory.heavy.intrinsic_flavors,
            xif=new_theory.xif,
            configs=new_operator.configs,
            debug=new_operator.debug,
            thresholds_config=tc,
            couplings=sc,
            interpol_dispatcher=bfd,
            n3lo_ad_variation=new_theory.n3lo_ad_variation,
        )

        with EKO.create(path) as builder:
            builder.load_cards(  # pylint: disable=E1101
                new_theory, new_operator
            ).build()

    def compute(self):
        """Run evolution and generate output operator.

        Two steps are applied sequentially:

        1. evolution is performed, computing the evolution operator in an
           internal flavor and x-space basis
        2. bases manipulations specified in the runcard are applied

        """
        with EKO.edit(self.path) as eko:
            # add all operators
            for final_scale, op in self.op_grid.compute().items():
                eko[float(final_scale)] = Operator.from_dict(op)
