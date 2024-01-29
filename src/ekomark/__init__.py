"""Benchmark package for eko."""

from eko import basis_rotation as br


def pdfname(pid_or_name):
    """Return pdf name."""
    if isinstance(pid_or_name, int):
        return br.flavor_basis_names[br.flavor_basis_pids.index(pid_or_name)]
    return pid_or_name
