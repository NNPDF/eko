# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import copy
import logging

import numpy as np

from eko import compatibility

from . import basis_rotation as br
from . import interpolation, msbar_masses
from .couplings import Couplings
from .evolution_operator.grid import OperatorGrid
from .output import Output
from .thresholds import ThresholdsAtlas

logger = logging.getLogger(__name__)


class Runner:
    """
    Represents a single input configuration.

    For details about the configuration, see :doc:`here </code/IO>`

    Parameters
    ----------
        setup : dict
            input configurations
    """

    banner = r"""
oooooooooooo oooo    oooo  \\ .oooooo.
`888'     `8 `888   .8P'  //////    `Y8b
 888          888  d8'   \\o\/////    888
 888oooo8     88888     \\\\/8/////   888
 888    "     888`88b.      888 ///   888
 888       o  888  `88b.    `88b //  d88'
o888ooooood8 o888o  o888o     `Y8bood8P'
"""

    def __init__(self, theory_card, operators_card):
        self.out = Output()

        new_theory = compatibility.update(theory_card)

        # Store inputs
        self._theory = new_theory

        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher.from_dict(operators_card)
        self.out.update(bfd.to_dict())
        # setup the Threshold path, compute masses if necessary
        masses = None
        if new_theory["HQ"] == "MSBAR":
            masses = msbar_masses.compute(new_theory)
        tc = ThresholdsAtlas.from_dict(new_theory, masses=masses)

        self.out["q2_ref"] = float(tc.q2_ref)
        # strong coupling
        sc = Couplings.from_dict(new_theory, masses=masses)
        # setup operator grid
        self.op_grid = OperatorGrid.from_dict(
            new_theory,
            operators_card,
            tc,
            sc,
            bfd,
        )
        self.out["inputgrid"] = bfd.xgrid_raw
        self.out["targetgrid"] = bfd.xgrid_raw
        self.post_process = dict(
            inputgrid=operators_card.get("inputgrid", bfd.xgrid_raw),
            targetgrid=operators_card.get("targetgrid", bfd.xgrid_raw),
            inputbasis=operators_card.get("inputbasis"),
            targetbasis=operators_card.get("targetbasis"),
        )

    def get_output(self):
        """
        Collects all data for output (to run the evolution)

        Returns
        -------
            ret : eko.output.Output
                output instance
        """
        # add all operators
        Q2grid = {}
        self.out["inputpids"] = br.flavor_basis_pids
        self.out["targetpids"] = br.flavor_basis_pids
        self.out["inputgrid"] = self.out["interpolation_xgrid"]
        self.out["targetgrid"] = self.out["interpolation_xgrid"]
        for final_scale, op in self.op_grid.compute().items():
            Q2grid[float(final_scale)] = op
        self.out["Q2grid"] = Q2grid
        # reshape xgrid
        inputgrid = (
            self.post_process["inputgrid"]
            if self.post_process["inputgrid"] is not self.out["interpolation_xgrid"]
            else None
        )
        targetgrid = (
            self.post_process["targetgrid"]
            if self.post_process["targetgrid"] is not self.out["interpolation_xgrid"]
            else None
        )
        if inputgrid is not None or targetgrid is not None:
            self.out.xgrid_reshape(targetgrid=targetgrid, inputgrid=inputgrid)

        # reshape flavors
        inputbasis = self.post_process["inputbasis"]
        if inputbasis is not None:
            inputbasis = np.array(inputbasis)
        targetbasis = self.post_process["targetbasis"]
        if targetbasis is not None:
            targetbasis = np.array(targetbasis)
        if inputbasis is not None or targetbasis is not None:
            self.out.flavor_reshape(targetbasis=targetbasis, inputbasis=inputbasis)
        return copy.deepcopy(self.out)
