# -*- coding: utf-8 -*-
"""
This file contains the main loop for the DGLAP calculations.
"""

def run_dglap(setup):
    """This function takes a DGLAP theory configuration dictionary
    and performs the solution of the DGLAP equations.

    Args:
        setup (dict): a dictionary with the theory parameters for the DGLAP

    Returns:
        kernel (array): a tensor operator in x-space.
    """

    # print theory id setup
    print(setup)

    """TODO:
    Points to be implemented:
        - allocate splittings, running
        - solve DGLAP in N-space
        - perform Mellin inverse
        - return the kernel operator in x-space
    """
    return 0