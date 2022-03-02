# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import yaml

here = pathlib.Path(__file__).parent
# Expose the default template
with open(here / "templatePDF.info", "r", encoding="utf-8") as o:
    template_info = yaml.safe_load(o)
with open(here / "Toy.info", "r", encoding="utf-8") as t:
    Toy_info = yaml.safe_load(t)


def load_info_from_file(pdfset_name):
    """
    Load the info file from a parent pdf.

    Parameters
    ----------
        pdfset_name : str
            parent pdf name

    Returns
    -------
        dict :
            info dictionary
    """
    import lhapdf  # pylint: disable=import-error, import-outside-toplevel

    src = pathlib.Path(lhapdf.paths()[0]) / pdfset_name
    with open(src / ("%s.info" % (pdfset_name)), "r", encoding="utf-8") as o:
        info = yaml.safe_load(o)
    return info


def load_blocks_from_file(pdfset_name, member):
    """
    Load a pdf from a parent pdf.

    Parameters
    ----------
        pdfset_name : str
            parent pdf name
        member : int
            pdf member

    Returns
    -------
        str :
            head of member file
        list(dict) :
            pdf blocks of data

    """
    import lhapdf  # pylint: disable=import-error, import-outside-toplevel

    pdf = lhapdf.mkPDF(pdfset_name, member)
    src = pathlib.Path(lhapdf.paths()[0]) / pdfset_name
    # read actual file
    cnt = []
    with open(
        src / ("%s_%04d.dat" % (pdfset_name, pdf.memberID)), "r", encoding="utf-8"
    ) as o:
        cnt = o.readlines()
    # file head
    head = cnt[0]
    head_section = cnt.index("---\n")
    blocks = []
    while head_section < len(cnt) - 1:
        # section head
        next_head_section = cnt.index("---\n", head_section + 1)
        # determine participating pids
        xgrid = np.array(cnt[head_section + 1].strip().split(" "), dtype=np.float_)
        Qgrid = np.array(cnt[head_section + 2].strip().split(" "), dtype=np.float_)
        pids = np.array(cnt[head_section + 3].strip().split(" "), dtype=np.int_)
        # data
        data = []
        for l in cnt[head_section + 4 : next_head_section]:
            elems = np.fromstring(l.strip(), sep=" ")
            data.append(elems)
        Q2grid = [el * el for el in Qgrid]
        blocks.append(dict(xgrid=xgrid, Q2grid=Q2grid, pids=pids, data=np.array(data)))
        # iterate
        head_section = next_head_section
    return head, blocks
