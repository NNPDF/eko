"""PDF set writer."""

import io
import pathlib
import re
from typing import Optional

import numpy as np
import yaml


def list_to_str(ls, fmt="%.6e"):
    """Convert a list of numbers to a string.

    Parameters
    ----------
    ls : list(float)
        list
    fmt : str
        format string

    Returns
    -------
    str :
        final string
    """
    return " ".join([fmt % x for x in ls])


def array_to_str(ar):
    """Convert an array of numbers to a string.

    Parameters
    ----------
    ar : list(list(float))
        array

    Returns
    -------
    str :
        final string
    """
    table = ""
    for line in ar:
        table += f"{line[0]:.8e} " + list_to_str(line[1:], fmt="%.8e") + "\n"
    return table


def dump_blocks(
    name: str, member: int, blocks: list[dict], pdf_type: Optional[str] = None
) -> pathlib.Path:
    """Write LHAPDF data file.

    Parameters
    ----------
    name : str or os.PathLike
        target name or path
    member : int
        PDF member
    blocks : list(dict)
        pdf blocks of data
    pdf_type : str
        PdfType to be copied in the head of member files

    Returns
    -------
        pathlib.Path : target file
    """
    path_name = pathlib.Path(name)
    if path_name.suffix == ".dat":
        target = path_name
    else:
        target = path_name / f"{path_name.stem}_{member:04d}.dat"
    target.parent.mkdir(exist_ok=True)
    with open(target, "w", encoding="utf-8") as o:
        if pdf_type is None:
            if member == 0:
                o.write("PdfType: central\n")
            else:
                o.write("PdfType: replica\n")
        else:
            o.write(pdf_type)
        o.write("Format: lhagrid1\n---\n")
        for b in blocks:
            o.write(list_to_str(b["xgrid"]) + "\n")
            o.write(list_to_str(list(np.sqrt(b["mu2grid"]))) + "\n")
            o.write(list_to_str(b["pids"], "%d") + "\n")
            o.write(array_to_str(b["data"]))
            o.write("---\n")
    return target


def dump_info(name: str, info: dict) -> pathlib.Path:
    """Write LHAPDF info file.

    NOTE: Since LHAPDF info files are not truly yaml files,
    we have to use a slightly more complicated function to
    dump the info file.

    Parameters
    ----------
    name : str or os.Pathlike
        target name or path
    info : dict
        info dictionary

    Returns
    -------
        pathlib.Path : target file
    """
    path_name = pathlib.Path(name)
    if path_name.suffix == ".info":
        target = path_name
    else:
        target = path_name / f"{path_name.stem}.info"
    target.parent.mkdir(exist_ok=True)
    # write on string stream to capture output
    stream = io.StringIO()
    yaml.safe_dump(info, stream, default_flow_style=True, width=100000, line_break="\n")
    cnt = stream.getvalue()
    # now insert some newlines for each key
    new_cnt = re.sub(r", ([A-Za-z_]+\d?):", r"\n\1:", cnt.strip()[1:-1])
    with open(target, "w", encoding="utf-8") as o:
        o.write(new_cnt)
    return target


def dump_set(name, info, member_blocks, pdf_type_list=None):
    """Dump a whole set.

    Parameters
    ----------
    name : str
        target name
    info : dict
        info dictionary
    member_blocks : list(list(dict))
        blocks for all members
    pdf_type : list(str)
        list of strings to be copied in the head of member files
    """
    dump_info(name, info)
    for mem, blocks in enumerate(member_blocks):
        if not isinstance(pdf_type_list, list) or len(pdf_type_list) == 0:
            dump_blocks(name, mem, blocks)
        else:
            dump_blocks(name, mem, blocks, pdf_type=pdf_type_list[mem])
