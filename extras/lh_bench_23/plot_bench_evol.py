from cfg import here, table_dir, xgrid
from utils import (
    compute_n3lo_avg_err,
    compute_n3lo_nnlo_diff,
    load_n3lo_tables,
    load_nnlo_table,
    plot_diff_to_nnlo,
    plot_pdfs,
)

USE_LINX = False
REL_DIFF = True
SCHEME = "VFNS"
SV = "central"

plot_dir = here / "plots_evol"
n3lo_table_dir = table_dir  # / SCHEME


# load tables
eko_dfs = load_n3lo_tables(
    n3lo_table_dir, SCHEME, sv="central", approx="EKO", rotate_to_evol=True
)
fhmruvv_dfs = load_n3lo_tables(
    n3lo_table_dir, SCHEME, sv="central", approx="FHMRUVV", rotate_to_evol=True
)
nnlo_central = load_nnlo_table(table_dir, SCHEME, SV, rotate_to_evol=True)

# compute avg and std
eko_res = compute_n3lo_avg_err(eko_dfs)
fhmruvv_res = compute_n3lo_avg_err(fhmruvv_dfs)

n3lo_dfs = [
    (eko_res, "aN3LO EKO"),
    (fhmruvv_res, "aN3LO FHMRUVV"),
]

# absolute plots
plot_pdfs(xgrid, n3lo_dfs, nnlo_central, SCHEME, USE_LINX, plot_dir)

# relative, absolute diff plots
eko_diff = compute_n3lo_nnlo_diff(eko_res, nnlo_central, REL_DIFF)
fhmruvv_diff = compute_n3lo_nnlo_diff(fhmruvv_res, nnlo_central, REL_DIFF)
n3lo_dfs = [
    (eko_diff, "aN3LO EKO"),
    (fhmruvv_diff, "aN3LO FHMRUVV"),
]

plot_diff_to_nnlo(xgrid, n3lo_dfs, SCHEME, USE_LINX, plot_dir, REL_DIFF)
