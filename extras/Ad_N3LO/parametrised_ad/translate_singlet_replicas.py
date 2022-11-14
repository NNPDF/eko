import pathlib
import eko
from unicodedata import name

fps = pathlib.Path(__file__).parent.glob("*3*nf*.c")

eko_path=eko.__path__

for fp in fps:
    with open(fp, encoding="utf-8") as oi:
        nls = []
        # translate
        for l in oi:
            l = l.replace("Sigma_Summation_Objects_Private_MyPower","np.power")
            l = l.replace("Power", "np.power")
            l = l.replace("Log", "np.log")
            l = l.replace("ln2", "np.log(2)")
            l = l.replace("EulerGamma", "np.euler_gamma")
            l = l.replace("S(1.,n)", "S1")
            l = l.replace("S(2.,n)", "S2")
            l = l.replace("S(3.,n)", "S3")
            nls.append(l)

    # output
    gamma_name, nf = str(fp).split('/')[-1].split('3')
    nf = nf[:-2]
    if "ps" in nf:
        nf = nf[2:]
    if gamma_name =="gqq":
        gamma_name = "gqqPS"
    eko_path = pathlib.Path(f"{eko_path}/anomalous_dimensions/as4/tmp")
    eko_path.mkdir(exist_ok=True)
    with open(f"{eko_path}/{gamma_name}.py", "a", encoding="utf-8") as oo:
        # oo.write(
        #     "# -*- coding: utf-8 -*-\n"
        # )
        # oo.write("import numba as nb\n")
        # oo.write("import numpy as np\n")
        oo.write("\n\n")
        oo.write('@nb.njit(cache=True)\n')
        name = gamma_name[1:]
        if "PS" in name:
           name = "ps"
        oo.write(f"def gamma_{name}_{nf}(n, sx, variation):\n")
        if "S1" in nls[0]:
            oo.write("    S1 = sx[0][0]\n")
        if "S2" in nls[0]:
            oo.write("    S2 = sx[1][0]\n")
        if "S3" in nls[0]:
            oo.write("    S3 = sx[2][0]\n")
        if "S4" in nls[0]:
            oo.write("    S4 = sx[3][0]\n")
        if "S5" in nls[0]:
            oo.write("    S5 = sx[4][0]\n")
        if "S3m2" in nls[1]:
            oo.write("    S3m2 = (-(((-1 + 2 * n) * (1 - n + n**2))/((-1 + n)**3 * n**3)) + S3)/n\n")
        if "S2m2" in nls[1]:
            oo.write("    S2m2 = ((-1 + 2 * n - 2 * n**2)/((-1 + n)**2 * n**2) + S2)/n\n")
        if "S2m2" in nls[1]:
            oo.write("    S1m2 = ((1 - 2 * n)/((-1 + n) * n) + S1)/n\n")
        oo.write(f"    common = {nls[0]}")
        oo.write("    if variation == 1:\n")
        oo.write(f"        fit = {nls[2]}")
        for idx, variation in enumerate(nls[3:]):
            oo.write(f"    elif variation == {idx+2}:\n")
            oo.write(f"        fit = {variation}")
        oo.write("    else:\n")
        oo.write(f"        fit = {nls[1]}")
        oo.write("    return common + fit")
        oo.write("\n\n")
