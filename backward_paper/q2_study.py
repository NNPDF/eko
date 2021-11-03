# -*- coding: utf-8 -*-
"""
This script compute an EKO at different Q2 and show the differences from pch and fitted charm
"""
import numpy as np

from banana.data import cartesian_product
from ekomark.data import operators
from eko.interpolation import make_grid

from runner import BackwardPaperRunner


class Q2gridRunner(BackwardPaperRunner):
    """
    Evolve pch and fitted charm to different Q2
    with different soltuion methods.
    """

    external = "inputpdf"

    base_theory = {
        "Qref": [9.1187600e01],
        "alphas": [0.1180024],
        "mc": [1.51],
        "mb": [4.92],
        "mt": [172.5],
        "kcThr": [1.0],
        "kbThr": [1.0],
        "ktThr": [1.0],
        "PTO": [3],
        "IC": [1],
        "IB": [1],
        "ModEv": ["TRN"],
    }
    base_operator = {
        "interpolation_xgrid": [make_grid(20, 30, x_min=1e-4).tolist()],
        "interpolation_polynomial_degree": [4],
        # "backward_inversion": ["exact"],
        "ev_op_iterations": [1],
    }

    def evolve_different_q2(self, pdf_name, q_low=1.5, q_grid=[1.65, 10.0, 100.0]):
        """
        Evolve pdf at differnt Q2 values

        Parameters
        ----------
            pdf_name: str
                PDF name
            q_low: float
                initial Q scale
            q_grid: list
                final Q scales
        """

        self.fig_name = pdf_name

        theory_updates = self.base_theory.copy()
        theory_updates["Q0"] = [q_low]

        operator_updates = self.base_operator.copy()
        operator_updates["Q2grid"] = [list(np.power(q_grid, 2))]
        self.run(
            cartesian_product(theory_updates),
            operators.build((operator_updates)),
            [pdf_name],
            use_replicas=True,
        )


if __name__ == "__main__":

    myrunner = Q2gridRunner()

    # Pch evolution
    name = "NNPDF40_nnlo_pch_as_01180"
    q_grid = [1.65, 10, 100]
    myrunner.evolve_different_q2(name, q_grid=q_grid)

    # Fitted charm evolution
    # pdf_names = {
    #     "NNPDF40_nnlo_as_01180": [10, 100],  # NNLO, fitted charm
    #     # "210701-n3fit-data-014": [10, 100],  # NNLO, fitted charm + EMC F2c
    # }
    # for name, q_grid in pdf_names.items():
    #     myrunner.evolve_different_q2(name, q_low=1.65, q_grid=q_grid)
