import copy
import pathlib

import numpy as np
import pytest
from banana.utils import lhapdf_path

from ekobox import genpdf

test_pdf = pathlib.Path(__file__).parents[1] / "fakepdf"

lhapdf = pytest.importorskip("lhapdf")


@pytest.mark.isolated
def benchmark_dump_info(tmp_path, cd):
    with cd(tmp_path):
        with lhapdf_path(test_pdf):
            info = genpdf.load.load_info_from_file("myCT14llo_NF3")
        info["SetDesc"] = "What ever I like"
        genpdf.export.dump_info("new_pdf", info)
        with lhapdf_path(tmp_path):
            info2 = genpdf.load.load_info_from_file("new_pdf")
            # my field is new
            assert info2["SetDesc"] == "What ever I like"
            # all the others are as before
            for k, v in info.items():
                if k == "SetDesc":
                    continue
                assert v == info2[k]


@pytest.mark.isolated
def benchmark_dump_blocks(tmp_path, cd):
    with cd(tmp_path):
        with lhapdf_path(test_pdf):
            info = genpdf.load.load_info_from_file("myCT14llo_NF3")
            blocks = genpdf.load.load_blocks_from_file("myCT14llo_NF3", 0)[1]
        new_blocks = copy.deepcopy(blocks)
        new_blocks[0]["xgrid"][0] = 1e-10
        heads = ["PdfType: debug\n"]
        genpdf.export.dump_set("new_pdf", info, [new_blocks], pdf_type_list=heads)
        with lhapdf_path(tmp_path):
            dat = genpdf.load.load_blocks_from_file("new_pdf", 0)
            head = dat[0]
            assert head == "PdfType: debug\n"
            blocks2 = dat[1]
            assert len(blocks) == len(blocks2)
            # my field is new
            np.testing.assert_allclose(blocks2[0]["xgrid"][0], 1e-10)
            # all the others are as before
            for k, v in blocks[0].items():
                if k == "xgrid":
                    continue
                np.testing.assert_allclose(v, blocks2[0][k])
            _pdf = lhapdf.mkPDF("new_pdf", 0)
        for x in new_blocks[0]["xgrid"][1:-1]:
            for mu in new_blocks[0]["mugrid"]:
                data_from_block = new_blocks[0]["data"][
                    new_blocks[0]["mugrid"].index(mu)
                    + len(new_blocks[0]["mugrid"])
                    * list(new_blocks[0]["xgrid"]).index(x)
                ][6]
                data_from_pdf = _pdf.xfxQ2(21, x, mu**2)
                np.testing.assert_allclose(data_from_block, data_from_pdf)
