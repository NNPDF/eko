# -*- coding: utf-8 -*-
"""
    This file contains the main application class of eko
"""
import copy
import logging

import numpy as np

from . import basis_rotation as br
from . import interpolation
from .evolution_operator.grid import OperatorGrid
from .output import Output
from .strong_coupling import StrongCoupling
from .thresholds import ThresholdsAtlas
from .msbar_masses import evolve_msbar_mass

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
        # setup the Threshold path, compute masses if necesssary
        masses = None
        if theory_card["HQ"] == "MSBAR":
            masses = compute_msbar_mass(theory_card)
        tc = ThresholdsAtlas.from_dict(theory_card, masses=masses)

        self.out["q2_ref"] = float(tc.q2_ref)
        # strong coupling
        sc = StrongCoupling.from_dict(theory_card)
        # setup operator grid
        self.op_grid = OperatorGrid.from_dict(
            theory_card,
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


def compute_msbar_mass(theory_card):
    r"""
    Compute the :math:`\overline{MS}` masses solving the equation :math:`m_{\bar{MS}}(m) = m`

    Parameters
    ----------
        theory_card: dict
            theory run card

    Returns
    -------
        masses: list
            list of msbar masses
    """
    masses = np.full(3, np.inf)
    nf_active = 3
    config = {
        "as_ref": theory_card["alphas"],
        "q2a_ref": np.power(theory_card["Qref"], 2),
        "order": theory_card["PTO"],
        "fact_to_ren": theory_card["fact_to_ren_scale_ratio"] ** 2,
    }
    for qidx, hq in enumerate("cbt"):
        q2m_ref = np.power(theory_card[f"Qm{hq}"], 2)
        m2_ref = np.power(theory_card[f"m{hq}"], 2)
        # check if mass is already given at the pole
        if q2m_ref == m2_ref:
            masses[qidx] = m2_ref
            continue
        # if self.q2_ref > q2m_ref:
        #     raise ValueError("In MSBAR scheme Q0 must be lower than any Qm")
        # if q2m_ref > m2_ref:
        #     raise ValueError("In MSBAR scheme each heavy quark \
        #         mass reference scale must be smaller or equal than \
        #             the value of the mass itself"
        #     )
        config["thr_masses"] = masses
        masses[qidx] = evolve_msbar_mass(
            m2_ref, q2m_ref, qidx + nf_active, config=config
        )

    # Check the msbar ordering
    nf_active = 4
    for qidx, hq in enumerate("bt"):
        q2m_ref = np.power(theory_card[f"Qm{hq}"], 2)
        m2_ref = np.power(theory_card[f"m{hq}"], 2)
        m2_msbar = masses[qidx]
        config["thr_masses"] = masses
        # check that m_msbar_hq < msbar_hq+1 (m_msbar_hq)
        m2_test = evolve_msbar_mass(
            m2_ref, q2m_ref, qidx + nf_active, config=config, q2_to=m2_msbar
        )
        if m2_msbar > m2_test:
            raise ValueError(
                "The MSBAR masses do not preserve the correct ordering,\
                    check the inital reference values"
            )
    return masses
