import pathlib

import eko

eko_path = eko.__path__
fps = pathlib.Path(__file__).parent.glob("*.c")
out = {}
for fp in fps:
    with open(fp, encoding="utf-8") as oi:
        nls = []
        # translate
        for l in oi:
            l = l.replace("Sigma_Summation_Objects_Private_MyPower", "np.power")
            l = l.replace("Power", "np.power")
            l = l.replace("Log", "np.log")
            l = l.replace("ln2", "np.log(2)")
            l = l.replace("EulerGamma", "np.euler_gamma")
            l = l.replace("S(1.,n)", "S1")
            l = l.replace("S(2.,n)", "S2")
            l = l.replace("S(3.,n)", "S3")
            nls.append(l)
        out[fp.stem] = nls

    # output
    gamma_name = str(fp).split("/")[-1].split("3")[0]
    if gamma_name == "gqq":
        gamma_name = "gqqPS"
    with open(
        f"{eko_path}/anomalous_dimensions/as4/{gamma_name}_1.py", "w", encoding="utf-8"
    ) as oo:
        oo.write("# -*- coding: utf-8 -*-\n")
        oo.write('"""\n')
        oo.write(
            f"This module contains the anomalous dimension :math:`\\gamma_{gamma_name[1:]}^(3)`\n"
        )
        oo.write('"""\n')
        oo.write("import numba as nb\n")
        oo.write("import numpy as np\n")
        oo.write("from .....harmonics import cache as c\n")
        oo.write("\n\n")
        oo.write("@nb.njit(cache=True)\n")
        oo.write(f"def gamma_{gamma_name[1:]}_nf3(n, cache):\n")
        oo.write("\treturn ")
        for l in nls:
            oo.write(l)
        oo.write("\n\n")
