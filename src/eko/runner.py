# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import copy
import logging

import numpy as np

from . import basis_rotation as br
from . import compatibility, interpolation, msbar_masses
from .couplings import Couplings
from .evolution_operator.grid import OperatorGrid
from .output import EKO, manipulate
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
        self.out = EKO()

        new_theory, new_operators = compatibility.update(theory_card, operators_card)

        # Store inputs
        self._theory = new_theory

        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher.from_dict(new_operators)

        # setup the Threshold path, compute masses if necessary
        masses = None
        if new_theory["HQ"] == "MSBAR":
            masses = msbar_masses.compute(new_theory)
        tc = ThresholdsAtlas.from_dict(new_theory, masses=masses)

        # strong coupling
        sc = Couplings.from_dict(new_theory, masses=masses)
        # setup operator grid
        self.op_grid = OperatorGrid.from_dict(
            new_theory,
            new_operators,
            tc,
            sc,
            bfd,
        )

        rot = operators_card.get("rotations", {})
        self.post_process = dict(
            inputgrid=rot.get("inputgrid", bfd.xgrid_raw),
            targetgrid=rot.get("targetgrid", bfd.xgrid_raw),
            inputbasis=rot.get("inputbasis"),
            targetbasis=rot.get("targetbasis"),
        )

        if "rotations" not in operators_card:
            operators_card["rotations"] = {}
        operators_card["rotations"]["inputgrid"] = bfd.xgrid_raw
        operators_card["rotations"]["targetgrid"] = bfd.xgrid_raw

        self.out = EKO.from_dict(dict(Q0=np.sqrt(tc.q2_ref)) | operators_card)

    def get_output(self):
        """
        Collects all data for output (to run the evolution)

        Returns
        -------
            ret : eko.output.EKO
                output instance
        """
        # add all operators
        self.out.rotations.inputpids = np.array(br.flavor_basis_pids)
        self.out.rotations.targetpids = np.array(br.flavor_basis_pids)
        self.out.rotations.inputgrid = self.out.xgrid
        self.out.rotations.targetgrid = self.out.xgrid
        for final_scale, op in self.op_grid.compute().items():
            self.out[float(final_scale)] = op

        # reshape xgrid
        inputgrid = (
            self.post_process["inputgrid"]
            if self.post_process["inputgrid"] is not self.out.xgrid
            else None
        )
        targetgrid = (
            self.post_process["targetgrid"]
            if self.post_process["targetgrid"] is not self.out.xgrid
            else None
        )
        if inputgrid is not None or targetgrid is not None:
            manipulate.xgrid_reshape(
                self.out, targetgrid=targetgrid, inputgrid=inputgrid
            )

        # reshape flavors
        inputbasis = self.post_process["inputbasis"]
        if inputbasis is not None:
            inputbasis = np.array(inputbasis)
        targetbasis = self.post_process["targetbasis"]
        if targetbasis is not None:
            targetbasis = np.array(targetbasis)
        if inputbasis is not None or targetbasis is not None:
            manipulate.flavor_reshape(
                self.out, targetbasis=targetbasis, inputbasis=inputbasis
            )
        return copy.deepcopy(self.out)
