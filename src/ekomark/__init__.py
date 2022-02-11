# -*- coding: utf-8 -*-
"""
Additional package to benchmark eko.
"""
import pathlib

from banana import load_config

from eko import basis_rotation as br

from . import banana_cfg


def register(path):
    path = pathlib.Path(path)
    if path.is_file():
        path = path.parent

    banana_cfg.cfg = load_config(path)


def pdfname(pid_or_name):
    """Return pdf name"""
    if isinstance(pid_or_name, int):
        return br.flavor_basis_names[br.flavor_basis_pids.index(pid_or_name)]
    return pid_or_name
