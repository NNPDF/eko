# -*- coding: utf-8 -*-
"""
This script compute an EKO to evolve a PDF set under the charm thrshold replica by replica
"""
import numpy as np

from ekomark.data import operators

from eko.interpolation import make_lambert_grid, make_grid

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
        "ModEv": "EXA",
    }
    base_operator = {
        "interpolation_xgrid": [make_lambert_grid(60).tolist()],
        # "interpolation_xgrid": [make_lambert_grid(60, x_min=1e-2).tolist()],
        # "interpolation_xgrid": [make_grid(20, 30, x_min=1e-2).tolist()],
        # "interpolation_xgrid": [make_grid(50, 10, x_low=0.8, x_min=1e-2).tolist()],
        # "interpolation_xgrid": [make_grid(30, 10, x_low=0.6, x_min=1e-2).tolist()],
        # "interpolation_polynomial_degree": [1],
        "backward_inversion": ["exact"],
        # "ev_op_iterations": [1],
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
        self.fig_name = None
        if return_to_Q0:
            self.return_to_Q0 = return_to_Q0
            self.fig_name = f"back_forth_{pdf_name}_{q_low}_{q_high}_{operator_updates['backward_inversion'][0]}"  # pylint: disable=line-too-long
            theory_updates["IC"] = 1
            theory_updates["IB"] = 1
            # self.rotate_to_evolution_basis = True
            # self.plot_pdfs = ["S", "g", "V"]
            # self.plot_pdfs = [21, -5, 5]

            operator_updates["Q2grid"] = [[q_low ** 2], [q_high ** 2], [q_high ** 2]]
            theory_updates["Q0"] = [q_high, q_low, q_high]
            return self.run_back_forth(
                [theory_updates],
                operators.build((operator_updates)),
                [pdf_name],
            )

        return self.doit(pdf_name, operator_updates, theory_updates, q_high, q_low)

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

    def evolve_backward_mass_variation(
        self, pdf_name, q_high=1.65, charm_mass=1.51, epsilon=0.01
    ):
        """
        Backward evolution changing also the charm mass value
        (the final Q scale is set to charm mass - epsilon)

        Parameters
        ----------
            pdf_name: str
                PDF name
            q_high: float
                initial Q scale
            charm_mass: float
                charm mass
        """
        operator_updates = self.base_operator.copy()
        theory_updates = self.base_theory.copy()
        theory_updates["mc"] = charm_mass
        q_low = charm_mass - epsilon
        self.fig_name = None
        return self.doit(pdf_name, operator_updates, theory_updates, q_high, q_low)


if __name__ == "__main__":

    myrunner = BackwardRunner()

    # Basic checks
    # pdf_names = [
    #     "NNPDF40_nnlo_as_01180",  # NNLO, fitted charm
    #     # "NNPDF40_nnlo_pch_as_01180",  # NNLO, perturbative charm
    # ]
    # for name in pdf_names:
    #     # Simple inversion
    #     myrunner.evolve_backward(name)
    #     # Test beclow above thr
    #     myrunner.evolve_above_below_thr(name)
    #    # Test exapanded/exact
    #    myrunner.evolve_exact_expanded(name)

    # # Test perturbarive B
    # pdf_name = "NNPDF40_nnlo_as_01180"
    # myrunner.evolve_above_below_thr(pdf_name, q_high=5, heavy_quark="b")

    # # Test EKO back and forth
    # myrunner.evolve_backward(pdf_name, q_high=5, q_low=30, return_to_Q0=True)
    # myrunner.evolve_backward(pdf_name, q_low=5, q_high=4.91, return_to_Q0=True)
    # myrunner.evolve_backward(pdf_name, q_low=4.91, q_high=5, return_to_Q0=True)

    # # pch vs ic forward matching
    # # q_low = 1.51 to compare with Silvia's plot
    # pdf_name = "NNPDF40_nnlo_pch_as_01180"
    # myrunner.evolve_backward(pdf_name, q_low=1.51, q_high=1.5, return_to_Q0=False)
    # pdf_name = "NNPDF40_nnlo_as_01180"
    # myrunner.evolve_backward(pdf_name, q_low=1.5101, q_high=1.65, return_to_Q0=False)

    # # charm mass variations, also used for momentum fractions
    # mass_variations = {
    #     1.38: "211103-ern-001",  # NNPDF40 baseline with mc=1.38
    #     1.64: "211103-ern-002",  # NNPDF40 baseline with mc=1.64
    #     1.51: "NNPDF40_nnlo_as_01180",  # NNLO, fitted charm
    #     1.51: "NNPDF40_nnlo_as_01180_EMC", # NNPDF40 baseline, fitted charm + EMC F2c
    # }
    # for c_m, pdf_name in mass_variations.items():
    #     myrunner.evolve_backward_mass_variation(pdf_name, q_high=1.65, charm_mass=c_m, epsilon=1e-8)

    # mass variation perturbative charm, forward evolution
    # mass_variations_pch = {
    #     1.38: "211103-ern-003",  # NNPDF40 baseline with mc=1.38
    #     1.64: "211103-ern-004",  # NNPDF40 baseline with mc=1.64
    #     1.51: "NNPDF40_nnlo_pch_as_01180",  # NNLO, fitted charm
    # }
    # for c_m, pdf_name in mass_variations_pch.items():
    #     myrunner.evolve_backward_mass_variation(pdf_name, q_high=c_m-1e-8, charm_mass=c_m, epsilon=c_m-1.65)

    # # dataset_variations
    # pdf_names =[
    #     "NNPDF40_nnlo_as_01180_EMC", # NNPDF40 baseline, fitted charm + EMC F2c
    #     "NNPDF40_nnlo_as_01180_DIS_only",  # NNPDF40 baseline DIS only
    #     "NNPDF40_nnlo_as_01180_collider_only", # NNPDF40 baseline collider only
    #     "NNPDF40_nnlo_as_01180_noLHCb", # NNPDF40 baseline no LHCb
    #     "NNPDF40_nnlo_as_01180",  # NNLO, fitted charm
    # ]
    # for name in pdf_names:
    #     myrunner.evolve_backward(name)

    # # Evolve above thr for RW fits inclding LHCb Z+c
    # pdf_names = [
    #     "NNPDF40_nnlo_as_0118_LHCbZc_uncorr",  # Uncorrelated sys
    #     "NNPDF40_nnlo_as_0118_LHCbZc_corr",  # Correlated sys
    #     "NNPDF40_nnlo_as_0118_EMC_LHCbZc_uncorr",  # Uncorrelated sys
    #     "NNPDF40_nnlo_as_0118_EMC_LHCbZc_corr",  # Correlated sys
    #     "NNPDF40_nnlo_as_01180",
    #     "NNPDF40_nnlo_as_01180_EMC",
    # ]
    # for pdf_name in pdf_names:
    #     myrunner.evolve_backward(
    #         pdf_name, q_low=1.50, q_high=1.65, return_to_Q0=False
    #     )

    # Flavor basis vs 4.0 comparison
    # pdf_names = [
    #     "210701-n3fit-meth-013",
    #     "NNPDF40_nnlo_as_01180"
    # ]
    # for pdf_name in pdf_names:
    #     myrunner.evolve_backward(
    #         pdf_name, q_low=1.51-1e-8, q_high=1.65, return_to_Q0=False
    #     )
