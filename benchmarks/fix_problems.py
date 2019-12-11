import numpy as np
import time
import lhapdf
import eko.dglap as dglap
import eko.interpolation as interpolation

toy_xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])
nf = 5
pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118", 0)
q0 = 6.2897452
q02 = pow(q0, 2)
qgrid = np.array(
    [
        4.4756282e01,
        5.5191298e01,
        6.8637940e01,
        8.6115921e01,
        1.0903923e02,
        1.3938725e02,
        1.7995815e02,
        2.3474820e02,
        3.0952544e02,
        4.1270732e02,
        5.5671861e02,
    ]
)
q2grid = np.power(qgrid, 2)
alpha_s = pdf.alphasQ2(q02)


def get_xgrid(
    n_small,
    n_medium,
    n_big,
    small_threshold=1e-7,
    mid_threshold=0.1,
    big_threshold=0.95,
):
    xgrid_low = []
    xgrid_med = []
    xgrid_big = []
    if n_small:
        xgrid_low = interpolation.get_xgrid_linear_at_log(
            n_small, small_threshold, mid_threshold
        )
    if n_medium:
        xgrid_med = interpolation.get_xgrid_linear_at_id(
            n_medium, mid_threshold, big_threshold
        )
    if n_big:
        xgrid_big = 1.0 - interpolation.get_xgrid_linear_at_log(
            n_big, 1e-3, 1 - big_threshold
        )
    xgrid = np.unique(np.concatenate([xgrid_low, xgrid_med, xgrid_big]))
    return xgrid


def run_me(Qref, Q0, alphas, xgrid, q2grid, polynom_rank = 4):
    ret = dglap.run_dglap(
        {
            "PTO": 0,
            "alphas": alphas,
            "Qref": Qref,
            "Q0": Q0,
            "NfFF": nf,
            "xgrid_type": "custom",
            "xgrid": xgrid,
            "xgrid_polynom_rank": polynom_rank,
            "log_interpol": True,
            "targetgrid": toy_xgrid,
            "Q2grid": q2grid,
            "jobs": 12,
        }
    )
    return ret


def get_pdf(q2, grid):
    q_dict = []
    singlet_res = []
    gluon_res = []
    for x in grid:
        pdf_q = pdf.xfxQ2(x, q2)
        pdf_q["s"] = np.sum([pdf_q[i] + pdf_q[-i] for i in range(1, 6)])
        q_dict.append(pdf_q)
        singlet_res.append(pdf_q["s"] / x)
        gluon_res.append(pdf_q[21] / x)
    return q_dict, np.array(singlet_res), np.array(gluon_res)

def digest(numbers):
    return [f"{abs(100*i):2.2f}" for i in numbers]

def assess(new, ref):
    relative = (new-ref)/ref
    string = digest(relative)
    print(string)

def run_test(nl, nm, nh):
    print("")
    print(" > > > %%%%% > > > ")
    print("Running for a grid with:")
    print(f"{nl=} {nm=} {nh=}")
    xgrid = get_xgrid(nl, nm, nh)
    initial = time.time()
    op = run_me(q0, q0, alpha_s, xgrid, q2grid)
    final = time.time()
    print(f"Operator computed, took {final-initial} seconds")
    _, si, gi = get_pdf(q02, xgrid)
    all_s = []
    all_g = []
    for i, q2 in enumerate(q2grid):
        print(f"Checking {i}/{len(q2grid)} {q2=}")
        gg = op["operators"]["S_gg"][:,:,i]
        qg = op["operators"]["S_qg"][:,:,i]
        gq = op["operators"]["S_gq"][:,:,i]
        qq = op["operators"]["S_qq"][:,:,i]
        _, sf, gf = get_pdf(q2, toy_xgrid)
        new_s = np.dot(qg, gi) + np.dot(qq, si)
        new_g = np.dot(gg, gi) + np.dot(gq, si)
        assess(new_s, sf)
        assess(new_g, gf)
        all_s.append(sf)
        all_g.append(gf)
    return op, all_s, all_g, si, gi


if __name__ == "__main__":
    res = []
    res.append(run_test(35, 15, 0))
    res.append(run_test(70, 15, 0))
    res.append(run_test(70, 15, 3))
    res.append(run_test(70, 30, 5))
