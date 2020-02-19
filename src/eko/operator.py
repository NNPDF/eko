"""
    This module contains the operator class
"""

def _run_step():
    """
        Performs a single convolution step in a fixed parameter configuration
    """
    kernel_dispatcher = KernelDispatcher(basis_function_dispatcher, constants, nf, delta_t)
    # run non-singlet
    # run singlet
    return ret


def _run_step(
    setup, constants, basis_function_dispatcher, xgrid, nf, mu2init, mu2final, mu2step=None
):
    """
        Do a single convolution step in a fixed parameter configuration

        Parameters
        ----------
            setup: dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            basis_function_dispatcher : InterpolatorDispatcher
                basis functions
            xgrid : array
                output grid
            nf : int
                number of active flavours
            mu2init : float
                initial scale
            mu2final : float
                final scale

        Returns
        -------
            ret : dict
                output dictionary
    """
    logger.info("evolve [GeV^2] %e -> %e with nf=%d flavors", mu2init, mu2final, nf)
    # Setup the kernel dispatcher
    delta_t = alpha_s.get_evolution_params(setup, constants, nf, mu2init, mu2final,mu2step)
    kernel_dispatcher = KernelDispatcher(
        basis_function_dispatcher, constants, nf, delta_t
    )

    # run non-singlet
    ret_ns = _run_nonsinglet(kernel_dispatcher, xgrid)
    # run singlet
    ret_s = _run_singlet(kernel_dispatcher, xgrid)
    # join elements
    ret = utils.merge_dicts(ret_ns, ret_s)
    return ret






class Operator:
    """ Computed only upon calling compute """

    def __init__(self, q_from, q_to, delta_t, nf):
        self.qref = q_from
        self.q = q_to
        self.delta_t = delta_t
        self.nf = nf
        self._computed = False

    def compute(self):
        self._computed = True

    def __mul__(self, operator):
        """ Does the internal product of two operators """
        # Check that the operators are compatible

    def __add__(self, operator):
        """ Does the summation of two operators """
