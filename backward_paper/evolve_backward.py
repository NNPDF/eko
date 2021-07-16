# -*- coding: utf-8 -*-
"""
This script compute an EKO to evolve a PDF set under the charm thrshold replica by replica
"""
import numpy as np

from ekomark.data import operators
from eko.interpolation import make_grid

from runner import BackwardPaperRunner

pid_dict = {"c": 4, "b": 5, "t": 6}


class BackwardRunner(BackwardPaperRunner):
    """
    This class evolve a pdf below the charm thrshold and
    compare it with the initial pdf replica by replica.
    """

    external = "inputpdf"

    base_theory = {
        "Qref": 9.1187600e01,
        "alphas": 0.1180024,
        "mc": 1.51,
        "mb": 4.92,
        "mt": 172.5,
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "PTO": 2,
        "IC": 1,
        "IB": 1,
    }
    base_operator = {
        # "interpolation_xgrid": [np.linspace(1e-1,1,50)],
        # "interpolation_xgrid": [make_grid(30,30, x_min=1e-3).tolist()],
        # "interpolation_polynomial_degree": 1,
        "backward_inversion": ["exact"]  # ["expanded","exact"],
    }

    def doit(
        self, pdf_name, operator_updates, theory_updates=None, q_high=None, q_low=None
    ):
        """Set common options and run"""
        if self.fig_name is None:
            self.fig_name = pdf_name
        if q_low is not None:
            operator_updates["Q2grid"] = [[q_low ** 2]]
        if theory_updates is None:
            theory_updates = self.base_theory.copy()
        if q_high is not None:
            theory_updates["Q0"] = q_high

        self.run(
            [theory_updates],
            operators.build((operator_updates)),
            [pdf_name],
            use_replicas=True,
        )

    def evolve_backward(self, pdf_name, q_high=1.65, q_low=1.5, return_to_Q0=False):
        """
        Base backward evolution

        Parameters
        ----------
            pdf_name: str
                PDF name
            q_high: float
                initial Q scale
            q_low: float
                final Q scale
            return_to_Q0: bool
                if True compute also the EKO back to test stability
        """
        operator_updates = self.base_operator.copy()
        theory_updates = self.base_theory.copy()
        if return_to_Q0:
            self.return_to_Q0 = return_to_Q0
            self.fig_name = f"back_forth_{pdf_name}_{q_low}_{q_high}"
            theory_updates["IC"] = 1
            theory_updates["IB"] = 1
            self.rotate_to_evolution_basis = True
            self.plot_pdfs = ["S", "g", "V"]
            # self.plot_pdfs = [21, -5, 5]

        self.doit(pdf_name, operator_updates, theory_updates, q_high, q_low)

    def evolve_above_below_thr(
        self, pdf_name, q_high=1.65, heavy_quark="c", epsilon=0.01
    ):
        """
        Comapare above and below the heavy quark threshold

        Parameters
        ----------
            pdf_name: str
                PDF name
            q_high: float
                initial Q scale
            heavy_quark: str
                heavy quark name
            epsilon: float
                distance from threshold
        """
        self.fig_name = f"compare_thr_{heavy_quark}_{pdf_name}"
        self.plot_pdfs = [-pid_dict[heavy_quark], pid_dict[heavy_quark]]

        operator_updates = self.base_operator.copy()
        thr_scale = (
            self.base_theory[f"m{heavy_quark}"] * self.base_theory[f"k{heavy_quark}Thr"]
        )
        operator_updates["Q2grid"] = [
            list(np.power([thr_scale + epsilon, thr_scale - epsilon], 2))
        ]
        self.doit(pdf_name, operator_updates, q_high=q_high)

    def evolve_exact_expanded(self, pdf_name, q_high=1.65, q_low=1.5):
        operator_updates = self.base_operator.copy()
        operator_updates["backward_inversion"] = ["exact", "expanded"]
        self.doit(pdf_name, operator_updates, q_high=q_high, q_low=q_low)


if __name__ == "__main__":

    myrunner = BackwardRunner()

    # Evolve below c threshold
    pdf_names = [
        "210629-n3fit-001",  # NNLO, fitted charm
        # "210629-theory-003",  # NNLO, perturbative charm
        # "210701-n3fit-data-014",  # NNLO, fitted charm + EMC F2c
        # "NNPDF31_nnlo_pch_as_0118"
    ]
    # for name in pdf_names:

    #     # # Simple inversion
    #     # myrunner.evolve_backward(name)

    #     # # Test beclow above thr
    #     myrunner.evolve_above_below_thr(name)

    #     # # Test exapanded/exact
    #     # myrunner.evolve_exact_expanded(name)

    # # Test perturbarive B
    pdf_name = "210629-n3fit-001"
    # pdf_name = "ToyLH"
    # myrunner.evolve_above_below_thr(pdf_name, q_high=5, heavy_quark="b")
    # myrunner.evolve_backward(pdf_name, q_low=5, q_high=4.91, return_to_Q0=True)
    # myrunner.evolve_backward(pdf_name, q_low=4.91, q_high=5, return_to_Q0=True)

    # # Test EKO back and forth
    myrunner.evolve_backward(pdf_name, q_high=30, q_low=100, return_to_Q0=True)
