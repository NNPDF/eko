from cfg import here, table_dir, xgrid
from utils import (
    compute_n3lo_avg_err,
    compute_n3lo_nnlo_diff,
    load_msht,
    load_n3lo_tables,
    load_nnlo_table,
    plot_diff_to_nnlo,
    plot_pdfs,
)

USE_LINX = False
REL_DIFF = True
SCHEME = "VFNS"
SV = "central"

plot_dir = here / "plots_msht"
n3lo_table_dir = table_dir  # / SCHEME
msht_table_dir = table_dir


# load tables
eko_dfs = load_n3lo_tables(n3lo_table_dir, SCHEME, approx="EKO")
fhmv_eko_dfs = load_n3lo_tables(n3lo_table_dir, SCHEME, approx="FHMV")
msht_dfs = load_msht(msht_table_dir, SCHEME, approx="MSHT")
fhmv_msht_dfs = load_msht(msht_table_dir, SCHEME, approx="FHMV")
nnlo_central = load_nnlo_table(table_dir, SCHEME, SV)

# compute avg and std
eko_res = compute_n3lo_avg_err(eko_dfs)
fhmv_eko_res = compute_n3lo_avg_err(fhmv_eko_dfs)
msht_res = compute_n3lo_avg_err(msht_dfs)
fhmv_msht_res = compute_n3lo_avg_err(fhmv_msht_dfs)
# eko_4mom_res = = compute_n3lo_avg_err(eko_dfs_4mom)

n3lo_dfs = [
    (eko_res, "EKO"),
    (fhmv_eko_res, "FHMV EKO"),
    (msht_res, "MSHT"),
    (fhmv_msht_res, "FHMV MSHT"),
    # (eko_4mom_res, "aN3LO EKO 4 mom"),
]

# PDFs plots
plot_pdfs(xgrid, n3lo_dfs, nnlo_central, SCHEME, USE_LINX, plot_dir)

# relative diff plots
eko_diff = compute_n3lo_nnlo_diff(eko_res, nnlo_central, REL_DIFF)
fhmv_eko_diff = compute_n3lo_nnlo_diff(fhmv_eko_res, nnlo_central, REL_DIFF)
msht_diff = compute_n3lo_nnlo_diff(msht_res, nnlo_central, REL_DIFF)
fhmv_msht_diff = compute_n3lo_nnlo_diff(fhmv_msht_res, nnlo_central, REL_DIFF)

n3lo_dfs = [
    (eko_diff, "EKO"),
    (fhmv_eko_diff, "FHMV EKO"),
    (msht_diff, "MSHT"),
    (fhmv_msht_diff, "FHMV MSHT"),
    # (eko_4mom_res, "aN3LO EKO 4 mom"),
]

# relative, absolute diff plots
plot_diff_to_nnlo(xgrid, n3lo_dfs, SCHEME, USE_LINX, plot_dir, REL_DIFF)
