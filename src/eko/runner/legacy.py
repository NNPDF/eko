"""Main application class of eko."""

import logging
from pathlib import Path
from typing import Union

from ekomark.data import update_runcards

from ..evolution_operator.grid import OperatorGrid
from ..io import EKO, Operator, runcards
from ..io.types import RawCard
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
        path: Path,
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
        new_theory, new_operator = update_runcards(theory_card, operators_card)

        # Store inputs
        self.path = path
        self._theory = new_theory

        # setup basis grid
        bfd = commons.interpolator(new_operator)

        # call explicitly iter to explain the static analyzer that is an
        # iterable
        tc = commons.atlas(new_theory, new_operator)

        # strong coupling
        cs = commons.couplings(new_theory, new_operator)  # setup operator grid

        # compute masses if required
        masses = runcards.masses(new_theory, new_operator.configs.evolution_method)

        self.op_grid = OperatorGrid(
            mu2grid=new_operator.evolgrid,
            order=new_theory.order,
            masses=masses,
            mass_scheme=new_theory.heavy.masses_scheme.value,
            thresholds_ratios=new_theory.heavy.squared_ratios,
            xif=new_theory.xif,
            configs=new_operator.configs,
            debug=new_operator.debug,
            atlas=tc,
            couplings=cs,
            interpol_dispatcher=bfd,
            n3lo_ad_variation=new_theory.n3lo_ad_variation,
            use_fhmruvv=new_theory.use_fhmruvv,
            matching_order=new_theory.matching_order,
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
            for ep, op in self.op_grid.compute().items():
                eko[ep] = Operator(**op)
