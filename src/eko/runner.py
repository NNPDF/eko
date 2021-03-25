# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import logging
import copy

from . import interpolation
from .output import Output
from .strong_coupling import StrongCoupling
from .thresholds import ThresholdsAtlas
from .operator.grid import OperatorGrid
from . import basis_rotation as br

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

    banner1 = """
EEEE  K  K   OOO
E     K K   O   O
EEE   KK    O   O
E     K K   O   O
EEEE  K  K   OOO
"""

    banner2 = """
EEEEEEE  KK  KK   OOOOO
EE       KK KK   OO   OO
EEEEE    KKKK    OO   OO
EE       KK KK   OO   OO
EEEEEEE  KK  KK   OOOOO"""

    banner3 = r""" # Varsity
 ________  ___  ____    ___    
|_   __  ||_  ||_  _| .'   `.  
  | |_ \_|  | |_/ /  /  .-.  \ 
  |  _| _   |  __'.  | |   | | 
 _| |__/ | _| |  \ \_\  `-'  / 
|________||____||____|`.___.'  
"""

    banner4 = r""" # Georgia11
`7MM\"""YMM  `7MMF' `YMM' .g8""8q.   
  MM    `7    MM   .M' .dP'    `YM. 
  MM   d      MM .d"   dM'      `MM 
  MMmmMM      MMMMM.   MM        MM 
  MM   Y  ,   MM  VMA  MM.      ,MP 
  MM     ,M   MM   `MM.`Mb.    ,dP' 
.JMMmmmmMMM .JMML.   MMb.`"bmmd"'   
"""

    # Roman
    banner5 = r"""
oooooooooooo oooo    oooo  \\ .oooooo.   
`888'     `8 `888   .8P'  //////    `Y8b  
 888          888  d8'   \\o\/////    888 
 888oooo8     88888[    \\\\/8/////   888 
 888    "     888`88b.      888 ///   888 
 888       o  888  `88b.    `88b //  d88' 
o888ooooood8 o888o  o888o     `Y8bood8P'  
"""

    def __init__(self, theory_card, operators_card):
        self.out = Output()

        # setup basis grid
        bfd = interpolation.InterpolatorDispatcher.from_dict(operators_card)
        self.out.update(bfd.to_dict())
        # FNS
        tc = ThresholdsAtlas.from_dict(theory_card)
        self.out["q2_ref"] = float(tc.q2_ref)
        # strong coupling
        sc = StrongCoupling.from_dict(theory_card, tc)
        # setup operator grid
        self.op_grid = OperatorGrid.from_dict(
            theory_card,
            operators_card,
            tc,
            sc,
            bfd,
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
        self.out["pids"] = br.flavor_basis_pids
        for final_scale, op in self.op_grid.compute().items():
            Q2grid[float(final_scale)] = op
        self.out["Q2grid"] = Q2grid
        return copy.deepcopy(self.out)
