# -*- coding: utf-8 -*-
"""
This file contains the main loop for the DGLAP calculations.
"""
import logging
from eko.constants import Constants

log = logging.getLogger(__name__)

def run_dglap(setup):
    """This function takes a DGLAP theory configuration dictionary
    and performs the solution of the DGLAP equations.

    Parameters:
    ----------
    setup: dict
        a dictionary with the theory parameters for the DGLAP

    Returns:
    -------
    kernel: array
        a tensor operator in x-space.
    """
    constants = Constants()

    # print theory id setup
    log.info(setup)
    # print constants
    log.info(constants)

#     TODO:
#     Points to be implemented:
#         - allocate splittings, running
#         - solve DGLAP in N-space
#         - perform Mellin inverse
#         - return the kernel operator in x-space
    return 0
