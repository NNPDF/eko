# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import pytest
from banana import toy
from banana.utils import lhapdf_path

from eko import basis_rotation as br
from ekobox import genpdf

test_pdf = pathlib.Path(__file__).parent / "genpdf"

lhapdf = pytest.importorskip("lhapdf")


@pytest.mark.isolated
def benchmark_genpdf_exceptions(tmp_path, cd):
    # using a wrong label and then a wrong parent pdf
    with cd(tmp_path):
        with pytest.raises(TypeError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions1",
                ["f"],
                {
                    21: lambda x, Q2: 3.0 * x * (1.0 - x),
                    2: lambda x, Q2: 4.0 * x * (1.0 - x),
                },
            )
        with pytest.raises(ValueError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions2",
                ["g"],
                10,
            )
        with pytest.raises(FileExistsError):
            genpdf.install_pdf("foo")

        with pytest.raises(TypeError):
            genpdf.generate_pdf("debug", [21], info_update=(10, 15, 20))


@pytest.mark.isolated
def benchmark_genpdf_no_parent_and_install(tmp_path, cd):
    with cd(tmp_path):
        d = tmp_path / "sub"
        d.mkdir()
        with lhapdf_path(d):
            genpdf.generate_pdf("test_genpdf_no_parent_and_install", [21], install=True)
        with lhapdf_path(d):
            pdf = lhapdf.mkPDF("test_genpdf_no_parent_and_install", 0)
            for x in [0.1, 0.2, 0.8]:
                for Q2 in [10.0, 20.0, 100.0]:
                    np.testing.assert_allclose(
                        pdf.xfxQ2(21, x, Q2), x * (1.0 - x), rtol=3e-5
                    )
                    np.testing.assert_allclose(pdf.xfxQ2(2, x, Q2), 0.0)


@pytest.mark.isolated
def benchmark_genpdf_toy(tmp_path, cd):
    with cd(tmp_path):
        toylh = toy.mkPDF("", 0)
        genpdf.generate_pdf(
            "test_genpdf_toy",
            [21],
            "toy",
            info_update={"NumFlavors": 25, "Debug": "Working"},
        )
        with lhapdf_path(tmp_path):
            # testing info updating
            info = genpdf.load.load_info_from_file("test_genpdf_toy")
            assert info["Debug"] == "Working"

            pdf = lhapdf.mkPDF("test_genpdf_toy", 0)
            for x in [0.1, 0.2, 0.5]:
                for Q2 in [10.0, 20.0, 100.0]:
                    np.testing.assert_allclose(
                        pdf.xfxQ2(21, x, Q2), toylh.xfxQ2(21, x, Q2), rtol=3e-5
                    )
                    np.testing.assert_allclose(pdf.xfxQ2(2, x, Q2), 0.0)


@pytest.mark.isolated
def benchmark_genpdf_parent_evolution_basis(tmp_path, cd):
    with cd(tmp_path):
        with lhapdf_path(test_pdf):
            CT14 = lhapdf.mkPDF("myCT14llo_NF3", 0)
            genpdf.generate_pdf(
                "test_genpdf_parent_evolution_basis", ["g"], "myCT14llo_NF3"
            )
        with lhapdf_path(tmp_path):
            pdf = lhapdf.mkPDF("test_genpdf_parent_evolution_basis", 0)
            for x in [0.1, 0.2, 0.5]:
                for Q2 in [10.0, 20.0, 100.0]:
                    np.testing.assert_allclose(
                        pdf.xfxQ2(21, x, Q2), CT14.xfxQ2(21, x, Q2), rtol=3e-5
                    )
                    np.testing.assert_allclose(pdf.xfxQ2(2, x, Q2), 0.0)


@pytest.mark.isolated
def benchmark_genpdf_dict(tmp_path, cd):
    with cd(tmp_path):
        genpdf.generate_pdf(
            "test_genpdf_dict",
            [21],
            {
                21: lambda x, Q2: 3.0 * x * (1.0 - x),
                2: lambda x, Q2: 4.0 * x * (1.0 - x),
            },
        )
        with lhapdf_path(tmp_path):
            pdf = lhapdf.mkPDF("test_genpdf_dict", 0)
            for x in [0.1, 0.2, 0.8]:
                for Q2 in [10.0, 20.0, 100.0]:
                    np.testing.assert_allclose(
                        pdf.xfxQ2(21, x, Q2), 3.0 * x * (1.0 - x), rtol=3e-5
                    )
                    # 2 is available, but not active
                    np.testing.assert_allclose(pdf.xfxQ2(2, x, Q2), 0.0)


@pytest.mark.isolated
def benchmark_genpdf_custom(tmp_path, cd):
    with cd(tmp_path):
        c = np.zeros_like(br.flavor_basis_pids, dtype=np.float_)
        c[br.flavor_basis_pids.index(1)] = 1.0
        c[br.flavor_basis_pids.index(2)] = 0.5
        genpdf.generate_pdf(
            "test_genpdf_custom",
            [c],
            {
                1: lambda x, Q2: 2.0 * x * (1.0 - x),
                2: lambda x, Q2: -4.0 * x * (1.0 - x),
            },
            info_update={"ForcePositive": 0},
        )
        with lhapdf_path(tmp_path):
            pdf = lhapdf.mkPDF("test_genpdf_custom", 0)
            for x in [0.1, 0.2, 0.8]:
                for Q2 in [10.0, 20.0, 100.0]:
                    # np.testing.assert_allclose(
                    #     pdf.xfxQ2(1, x, Q2),
                    #     0.0,err_msg=f"pid=1,x={x},Q2={Q2}"
                    # )
                    np.testing.assert_allclose(
                        pdf.xfxQ2(2, x, Q2), 0.0, err_msg=f"pid=2,x={x},Q2={Q2}"
                    )


@pytest.mark.isolated
def benchmark_genpdf_allflavors(tmp_path, cd):
    with cd(tmp_path):
        for setname in ("myMSTW2008nlo90cl", "myNNPDF31_nlo_as_0118"):
            with lhapdf_path(test_pdf):
                # load reference PDFs
                ref = {}
                ref_head = {}
                for mem in range(1 + 1):
                    ref[mem] = lhapdf.mkPDF(setname, mem)
                    ref_head[mem] = genpdf.load.load_blocks_from_file(setname, mem)[0]
                # filtering on all flavors
                genpdf.generate_pdf(
                    "test_genpdf_" + setname,
                    [21, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6],
                    setname,
                    members=True,
                )
            with lhapdf_path(tmp_path):
                for mem in range(1 + 1):
                    pdf = lhapdf.mkPDF("test_genpdf_" + setname, mem)
                    head = genpdf.load.load_blocks_from_file(
                        "test_genpdf_" + setname, mem
                    )[0]
                    assert head == ref_head[mem]
                    # testing for some pids, x and Q2 values
                    for pid in [21, 1, 2, -3, -5]:
                        for x in [0.1, 0.2, 0.8]:
                            for Q2 in [10.0, 20.0, 100.0]:
                                np.testing.assert_allclose(
                                    pdf.xfxQ2(pid, x, Q2),
                                    ref[mem].xfxQ2(pid, x, Q2),
                                    err_msg=f"mem={mem}, pid={pid}, x={x}, Q2={Q2}",
                                    rtol=1e6,
                                )
