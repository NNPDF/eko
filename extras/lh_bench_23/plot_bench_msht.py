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
eko_dfs = load_n3lo_tables(n3lo_table_dir, SCHEME, SV, approx="EKO")
fhmruvv_eko_dfs = load_n3lo_tables(n3lo_table_dir, SCHEME, SV, approx="FHMRUVV")
msht_dfs = load_msht(msht_table_dir, SCHEME, approx="MSHT")
fhmruvv_msht_dfs = load_msht(msht_table_dir, SCHEME, approx="FHMRUVV")
nnlo_central = load_nnlo_table(table_dir, SCHEME, SV)

# compute avg and std
eko_res = compute_n3lo_avg_err(eko_dfs)
fhmruvv_eko_res = compute_n3lo_avg_err(fhmruvv_eko_dfs)
msht_res = compute_n3lo_avg_err(msht_dfs)
fhmruvv_msht_res = compute_n3lo_avg_err(fhmruvv_msht_dfs)
# eko_4mom_res = = compute_n3lo_avg_err(eko_dfs_4mom)

n3lo_dfs = [
    (eko_res, "EKO"),
    (fhmruvv_eko_res, "FHMRUVV EKO"),
    (msht_res, "MSHT"),
    (fhmruvv_msht_res, "FHMRUVV MSHT"),
    # (eko_4mom_res, "aN3LO EKO 4 mom"),
]

# PDFs plots
plot_pdfs(xgrid, n3lo_dfs, nnlo_central, SCHEME, USE_LINX, plot_dir)

# relative diff plots
eko_diff = compute_n3lo_nnlo_diff(eko_res, nnlo_central, REL_DIFF)
fhmruvv_eko_diff = compute_n3lo_nnlo_diff(fhmruvv_eko_res, nnlo_central, REL_DIFF)
msht_diff = compute_n3lo_nnlo_diff(msht_res, nnlo_central, REL_DIFF)
fhmruvv_msht_diff = compute_n3lo_nnlo_diff(fhmruvv_msht_res, nnlo_central, REL_DIFF)

n3lo_dfs = [
    (eko_diff, "EKO"),
    (fhmruvv_eko_diff, "FHMRUVV EKO"),
    (msht_diff, "MSHT"),
    (fhmruvv_msht_diff, "FHMRUVV MSHT"),
    # (eko_4mom_res, "aN3LO EKO 4 mom"),
]

# relative, absolute diff plots
plot_diff_to_nnlo(xgrid, n3lo_dfs, SCHEME, USE_LINX, plot_dir, REL_DIFF)
