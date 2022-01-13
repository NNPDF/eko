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
from .strong_coupling import StrongCoupling, strong_coupling_mod_ev
from .thresholds import ThresholdsAtlas
from .msbar_masses import evolve_msbar_mass
from .evolution_operator.flavors import quark_names

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
        # setup the Threshold path, compute masses if necessary
        masses = None
        if theory_card["HQ"] == "MSBAR":
            masses = compute_msbar_mass(theory_card)
        tc = ThresholdsAtlas.from_dict(theory_card, masses=masses)

        self.out["q2_ref"] = float(tc.q2_ref)
        # strong coupling
        sc = StrongCoupling.from_dict(theory_card, masses=masses)
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
    Compute the |MS| masses solving the equation :math:`m_{\bar{MS}}(m) = m`

    Parameters
    ----------
        theory_card: dict
            theory run card

    Returns
    -------
        masses: list
            list of msbar masses squared
    """
    # TODO: sketch in the docs how the MSbar computation works with a figure.
    nfa_ref = theory_card["nfref"]

    q2_ref = np.power(theory_card["Qref"], 2)
    masses = np.concatenate((np.zeros(nfa_ref - 3), np.full(6 - nfa_ref, np.inf)))
    config = {
        "as_ref": theory_card["alphas"],
        "q2a_ref": q2_ref,
        "order": theory_card["PTO"],
        "fact_to_ren": theory_card["fact_to_ren_scale_ratio"] ** 2,
        "method": strong_coupling_mod_ev(theory_card["ModEv"]),
        "nfref": nfa_ref,
    }

    # First you need to look for the thr around the given as_ref
    heavy_quarks = quark_names[3:]
    hq_idxs = np.arange(0, 3)
    if nfa_ref > 4:
        heavy_quarks = reversed(heavy_quarks)
        hq_idxs = reversed(hq_idxs)

    # loop on heavy quarks and compute the msbar masses
    for q_idx, hq in zip(hq_idxs, heavy_quarks):
        q2m_ref = np.power(theory_card[f"Qm{hq}"], 2)
        m2_ref = np.power(theory_card[f"m{hq}"], 2)

        # check if mass is already given at the pole -> done
        if q2m_ref == m2_ref:
            masses[q_idx] = m2_ref
            continue

        # update the alphas thr scales
        config["thr_masses"] = masses
        nf_target = q_idx + 3
        shift = -1

        # check that alphas is given with a consistent number of flavors
        if q_idx + 4 == nfa_ref and q2m_ref > q2_ref:
            raise ValueError(
                f"In MSBAR scheme, Qm{hq} should be lower than Qref, \
                if alpha_s is given with nfref={nfa_ref} at scale Qref={q2_ref}"
            )
        if q_idx + 4 == nfa_ref + 1 and q2m_ref < q2_ref:
            raise ValueError(
                f"In MSBAR scheme, Qm{hq} should be greater than Qref, \
                if alpha_s is given with nfref={nfa_ref} at scale Qref={q2_ref}"
            )

        # check that for higher patches you do forward running
        # with consistent conditions
        if q_idx + 3 >= nfa_ref and q2m_ref >= m2_ref:
            raise ValueError(
                f"In MSBAR scheme, Qm{hq} should be lower than m{hq} \
                        if alpha_s is given with nfref={nfa_ref} at scale Qref={q2_ref}"
            )

        # check that for lower patches you do backward running
        # with consistent conditions
        if q_idx + 3 < nfa_ref:
            if q2m_ref < m2_ref:
                raise ValueError(
                    f"In MSBAR scheme, Qm{hq} should be greater than m{hq} \
                        if alpha_s is given with nfref={nfa_ref} at scale Qref={q2_ref}"
                )
            nf_target += 1
            shift = 1

        # if the initial condition is not in the target patch,
        # you need to evolve it until nf_target patch wall is reached:
        #   for backward you reach the higher, for forward the lower.
        # len(masses[q2m_ref > masses]) + 3 is the nf at the given reference scale
        if nf_target != len(masses[q2m_ref > masses]) + 3:
            q2_to = masses[q_idx + shift]
            m2_ref = evolve_msbar_mass(m2_ref, q2m_ref, config=config, q2_to=q2_to)
            q2m_ref = q2_to

        # now solve the RGE
        masses[q_idx] = evolve_msbar_mass(m2_ref, q2m_ref, nf_target, config=config)

    # Check the msbar ordering
    for m2_msbar, hq in zip(masses[:-1], quark_names[4:]):
        q2m_ref = np.power(theory_card[f"Qm{hq}"], 2)
        m2_ref = np.power(theory_card[f"m{hq}"], 2)
        config["thr_masses"] = masses
        # check that m_msbar_hq < msbar_hq+1 (m_msbar_hq)
        m2_test = evolve_msbar_mass(m2_ref, q2m_ref, config=config, q2_to=m2_msbar)
        if m2_msbar > m2_test:
            raise ValueError(
                "The MSBAR masses do not preserve the correct ordering,\
                    check the initial reference values"
            )
    return masses
