import numpy as np

from eko import anomalous_dimensions as ad
from eko.mellin import Path

NF = 5


class TimeSuite:
    def setup(self):
        ts = np.linspace(0.5, 1.0 - 1e-5, 100)
        logx = 0.8
        axis_offset = True
        self.ns = []
        for t in ts:
            self.ns.append(Path(t, logx, axis_offset).n)

    def time_as1_sing(self):
        for n in self.ns:
            ad.gamma_singlet(0, n, NF, p=False)

    def time_as2_sing(self):
        for n in self.ns:
            ad.gamma_singlet(1, n, NF, p=False)

    def time_as3_sing(self):
        for n in self.ns:
            ad.gamma_singlet(2, n, NF, p=False)
