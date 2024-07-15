"""
Benchmark to :cite:`Giele:2002hx` (LO + NLO) and :cite:`Dittmar:2005ed` (NNLO).
"""

import os

import numpy as np
import pytest
from banana import register

from eko.interpolation import lambertgrid
from ekomark.benchmark.runner import Runner

register(__file__)

_sqrt2 = float(np.sqrt(2.0))

base_theory = {
    "ModEv": "EXA",
    "Q0": _sqrt2,  # Eq. (30) :cite:`Giele:2002hx`, Eq. (4.53) :cite:`Dittmar:2005ed`
    "mc": _sqrt2,  # Eq. (34) :cite:`Giele:2002hx`, Eq. (4.56) :cite:`Dittmar:2005ed`
    "mb": 4.5,
    "mt": 175,
    "Qref": _sqrt2,  # Eq. (32) :cite:`Giele:2002hx`,Eq. (4.53) :cite:`Dittmar:2005ed`
    "alphas": 0.35,  # Eq. (4.55) :cite:`Dittmar:2005ed`
    "alphaqed": 0.007496,
    "QED": 0,
    "Qedref": 0.0,
}
"""Global theory settings"""

default_skip_pdfs = [22, -6, 6, "ph", "V35", "V24", "V15", "V8", "T35"]
# ffns_skip_pdfs = vfns_skip_pdfs.copy()
# ffns_skip_pdfs.extend([-5, 5, "T24"])


class LHA(Runner):
    """Globally set the external program to LHA."""

    def __init__(self):
        super().__init__()
        self.external = "LHA"
        self.theory = {}
        self.rotate_to_evolution_basis = True

    def plain_theory(self, pto):
        """Generate plain theories at given PTO.

        Parameters
        ----------
        pto : int
            perturbation order

        Returns
        -------
        list(dict)
            theory updates
        """
        th = self.theory.copy()
        th.update({"PTO": pto})
        return [th]

    def sv_theories(self, pto):
        """Generate scale variation theories.

        Parameters
        ----------
        pto : int
            perturbation order

        Returns
        -------
        list(dict)
            theory updates
        """
        low = self.theory.copy()
        low["PTO"] = pto
        low["XIF"] = 1.0 / _sqrt2
        low["ModSV"] = "exponentiated"
        high = self.theory.copy()
        high["PTO"] = pto
        high["XIF"] = _sqrt2
        high["ModSV"] = "exponentiated"
        return [high, low]

    @staticmethod
    def skip_pdfs(_theory):
        """Adjust skip_pdf by the used theory.

        Parameters
        ----------
        theory : dict
            theory update

        Returns
        -------
        list
            allowed PDFs in LHA
        """
        return default_skip_pdfs

    def run_lha(self, theory_updates):
        """Enforce operator grid and PDF.

        Parameters
        ----------
        theory_updates : list(dict)
            theory updates
        """
        self.run(
            theory_updates,
            [
                {
                    "mugrid": [100],
                    "ev_op_iterations": 10,
                    "interpolation_xgrid": lambertgrid(60).tolist(),
                }
            ],
            ["ToyLH"],
        )

    def run_plain(self, pto):
        """Run plain configuration."""
        self.run_lha(self.plain_theory(pto))

    def run_sv(self, pto):
        """Run scale variations."""
        self.run_lha(self.sv_theories(pto))


class BaseBenchmark:
    """Abstract common benchmark tasks."""

    def runner(self) -> LHA:
        """Runner to run."""
        raise NotImplementedError("runner method has to be overwritten!")

    def transformed_runner(self):
        """Prepare runner for benchmark setup"""
        r = self.runner()
        r.log_to_stdout = os.environ.get("EKO_LOG_STDOUT", False)
        return r

    @pytest.mark.lo
    def benchmark_plain_lo(self):
        self.transformed_runner().run_plain(0)

    @pytest.mark.nlo
    def benchmark_plain_nlo(self):
        self.transformed_runner().run_plain(1)

    @pytest.mark.nnlo
    def benchmark_plain_nnlo(self):
        self.transformed_runner().run_plain(2)

    @pytest.mark.nlo
    @pytest.mark.sv
    def benchmark_sv_nlo(self):
        self.transformed_runner().run_sv(1)

    @pytest.mark.nnlo
    @pytest.mark.sv
    def benchmark_sv_nnlo(self):
        self.transformed_runner().run_sv(2)


class VFNS(LHA):
    """Provide |VFNS| settings."""

    def __init__(self):
        super().__init__()
        self.theory = base_theory.copy()
        self.theory.update(
            {
                "FNS": "ZM-VFNS",  # ignored by eko, but needed by LHA_utils
                "kcThr": 1.0,
                "kbThr": 1.0,
                "ktThr": 1.0,
                "nf0": 3,
                "nfref": 3,
            }
        )


@pytest.mark.vfns
class BenchmarkVFNS(BaseBenchmark):
    def runner(self):
        return VFNS()


class FFNS(LHA):
    """Provide |FFNS| settings."""

    def __init__(self):
        super().__init__()
        self.theory = base_theory.copy()
        self.theory.update(
            {
                "FNS": "FFNS",  # ignored by eko, but needed by LHA_utils
                "NfFF": 4,
                "nfref": 4,
                "kcThr": 0.0,
                "kbThr": np.inf,
                "ktThr": np.inf,
            }
        )

    @staticmethod
    def skip_pdfs(theory):
        """Adjust skip_pdf by the used theory.

        Parameters
        ----------
        theory : dict
            theory update

        Returns
        -------
        list
            allowed PDFs in FFNS LHA
        """
        ffns_skip_pdfs = default_skip_pdfs.copy()
        # remove bottom
        ffns_skip_pdfs.extend([-5, 5, "T24"])
        # in NNLO also V8 gets removed
        if theory["PTO"] >= 2:
            ffns_skip_pdfs.remove("V8")
        return ffns_skip_pdfs


@pytest.mark.ffns
class BenchmarkFFNS(BaseBenchmark):
    def runner(self):
        return FFNS()


class FFNS_polarized(FFNS):
    def run_lha(self, theory_updates):
        """Enforce operator grid and PDF.

        Parameters
        ----------
        theory_updates : list(dict)
            theory updates
        """
        self.run(
            theory_updates,
            [
                {
                    "mugrid": [1e2],
                    "ev_op_iterations": 10,
                    "interpolation_xgrid": lambertgrid(60).tolist(),
                    "polarized": True,
                }
            ],
            ["ToyLH_polarized"],
        )


@pytest.mark.ffns_pol
class BenchmarkFFNS_polarized(BaseBenchmark):
    def runner(self):
        return FFNS_polarized()

    @pytest.mark.nnlo
    def benchmark_plain_nnlo(self):
        pass

    @pytest.mark.nnlo
    @pytest.mark.sv
    def benchmark_sv_nnlo(self):
        pass


class CommonRunner(VFNS):
    """Generic benchmark runner using the LHA |VFNS| settings."""

    def __init__(self, external):
        super().__init__()
        self.external = external
        self.sandbox = True

    def benchmark_sv(self, pto):
        """Run scale variations.

        Parameters
        ----------
        pto : int
            perturbation order
        """
        high, low = self.sv_theories(pto)

        # here we need to adjust also the
        # apfel initial nf, which can't
        # be accessed in other ways
        if self.external == "apfel":
            for sv_theory in [low, high]:
                sv_theory["kcThr"] = 1.0 + 1e-15
                sv_theory["nfref"] = 4
                sv_theory["EScaleVar"] = 0
        low["XIR"] = _sqrt2
        high["XIR"] = 1.0 / _sqrt2

        self.run_lha([low, high])
