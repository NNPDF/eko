import pathlib

import matplotlib.pyplot as plt
import numpy as np

from eko import interpolation

# setup
logxmin = -5
xgrid = np.logspace(logxmin, 0, 9)
polynomial_degree = 3
disp = interpolation.InterpolatorDispatcher(xgrid, polynomial_degree, True, False)

# show grid
xs = np.logspace(logxmin, 0, 150)
plt.xlabel("x")
plt.xscale("log")
plt.plot(xgrid, [0] * len(xgrid), "o", color="black")
# plot
for j in [0, 1, 4, 8]:
    ys = [disp[j](x) for x in xs]
    plt.plot(xs, ys, label=f"$p_{j}(x)$")

# global
plt.title("Example configuration")
plt.legend()

# output
plt.savefig(
    pathlib.Path(__file__).parent.parent / "img" / "interpolation-polynomials.png"
)
