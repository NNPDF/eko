# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import pytest
from banana.utils import lhapdf_path

from eko import basis_rotation as br
from ekobox import genpdf

test_pdf = pathlib.Path(__file__).parents[1] / "fakepdf"

lhapdf = pytest.importorskip("lhapdf")


@pytest.mark.isolated
def benchmark_flavors_pids_ct14(tmp_path, cd):
    with cd(tmp_path):
        # read the debug PDFs
        with lhapdf_path(test_pdf):
            info = genpdf.load.load_info_from_file("myCT14llo_NF3")
            blocks = genpdf.load.load_blocks_from_file("myCT14llo_NF3", 0)[1]
            pdf = lhapdf.mkPDF("myCT14llo_NF3", 0)
        # now extract the gluon
        new_blocks = genpdf.flavors.project(blocks, genpdf.flavors.pid_to_flavor([21]))
        info["Flavors"] = br.flavor_basis_pids
        info["NumFlavors"] = len(br.flavor_basis_pids)
        genpdf.export.dump_set("test_flavors_pids_ct14", info, [new_blocks])
        with lhapdf_path(tmp_path):
            gonly = lhapdf.mkPDF("test_flavors_pids_ct14", 0)
            # all quarks are 0
            for pid in [1, 2, -3]:
                for x in [1e-2, 0.1, 0.9]:
                    for Q2 in [10, 100]:
                        np.testing.assert_allclose(gonly.xfxQ2(pid, x, Q2), 0.0)
            # and the gluon in as before
            for x in [1e-2, 0.1, 0.9]:
                for Q2 in [10, 100]:
                    np.testing.assert_allclose(
                        pdf.xfxQ2(21, x, Q2), gonly.xfxQ2(21, x, Q2)
                    )


@pytest.mark.isolated
def benchmark_flavors_evol_ct14(tmp_path, cd):
    with cd(tmp_path):
        # read the debug PDFs
        with lhapdf_path(test_pdf):
            info = genpdf.load.load_info_from_file("myCT14llo_NF3")
            blocks = genpdf.load.load_blocks_from_file("myCT14llo_NF3", 0)[1]
            pdf = lhapdf.mkPDF("myCT14llo_NF3", 0)
        # now extract the gluon
        new_blocks = genpdf.flavors.project(
            blocks, genpdf.flavors.evol_to_flavor(["g"])
        )
        info["Flavors"] = br.flavor_basis_pids
        info["NumFlavors"] = len(br.flavor_basis_pids)
        genpdf.export.dump_set("test_flavors_evol_ct14", info, [new_blocks])
        with lhapdf_path(tmp_path):
            gonly = lhapdf.mkPDF("test_flavors_evol_ct14", 0)
            # all quarks are 0
            for pid in [1, 2, -3]:
                for x in [1e-2, 0.1, 0.9]:
                    for Q2 in [10, 100]:
                        np.testing.assert_allclose(gonly.xfxQ2(pid, x, Q2), 0.0)
            # and the gluon in as before
            for x in [1e-2, 0.1, 0.9]:
                for Q2 in [10, 100]:
                    np.testing.assert_allclose(
                        pdf.xfxQ2(21, x, Q2), gonly.xfxQ2(21, x, Q2)
                    )
