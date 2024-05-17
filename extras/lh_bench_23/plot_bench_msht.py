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
n3lo_table_dir = table_dir
msht_table_dir = table_dir


# load tables
eko_dfs = load_n3lo_tables(n3lo_table_dir, SCHEME, SV, approx="EKO")

fhmruvv_eko_dfs = load_n3lo_tables(n3lo_table_dir, SCHEME, SV, approx="FHMRUVV")
fhmruvv_msht_dfs = load_msht(msht_table_dir, SCHEME, approx="FHMRUVV")

msht_post_dfs = load_msht(msht_table_dir, SCHEME, approx="MSHTposterior")
msht_prior_dfs = load_msht(msht_table_dir, SCHEME, approx="MSHTprior")
nnlo_central = load_nnlo_table(table_dir, SCHEME, SV)

# compute avg and std
eko_res = compute_n3lo_avg_err(eko_dfs)
fhmruvv_eko_res = compute_n3lo_avg_err(fhmruvv_eko_dfs)
fhmruvv_msht_res = compute_n3lo_avg_err(fhmruvv_msht_dfs)
msht_post_res = compute_n3lo_avg_err(msht_post_dfs)
msht_prior_res = compute_n3lo_avg_err(msht_prior_dfs)

# compute average of FHMRUVV
fhmruvv_res = []
for a, b in zip(fhmruvv_msht_res, fhmruvv_eko_res):
    fhmruvv_res.append((a + b) / 2)

# PDFs plots
n3lo_dfs = [
    (fhmruvv_res, "FHMRUVV"),
    (msht_prior_res, "MSHT (prior)"),
    (msht_post_res, "MSHT (posterior)"),
    (eko_res, "NNPDF"),
]
plot_pdfs(xgrid, n3lo_dfs, nnlo_central, SCHEME, USE_LINX, plot_dir)

# relative, absolute diff plots
eko_diff = compute_n3lo_nnlo_diff(eko_res, nnlo_central, REL_DIFF)
fhmruvv_diff = compute_n3lo_nnlo_diff(fhmruvv_res, nnlo_central, REL_DIFF)
msht_prior_diff = compute_n3lo_nnlo_diff(msht_prior_res, nnlo_central, REL_DIFF)
msht_post_diff = compute_n3lo_nnlo_diff(msht_post_res, nnlo_central, REL_DIFF)

n3lo_dfs = [
    (fhmruvv_diff, "FHMRUVV"),
    (msht_prior_diff, "MSHT (prior)"),
    (msht_post_diff, "MSHT (posterior)"),
    (eko_diff, "NNPDF"),
]
plot_diff_to_nnlo(xgrid, n3lo_dfs, SCHEME, USE_LINX, plot_dir, REL_DIFF)
