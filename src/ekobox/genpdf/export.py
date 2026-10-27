"""PDF set writer."""

import io
import pathlib
import re
from typing import Optional

import yaml

from .parser import LhapdfDataFile


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


def dump_set(
    name: str,
    info: dict,
    member_files: list[LhapdfDataFile],
    header_list: Optional[list[dict[str, str]]] = None,
) -> None:
    """Dump a whole set.

    Parameters
    ----------
    name :
        target name
    info :
        info dictionary
    member_files :
        blocks for all members
    header_list :
        list of optional additional headers to be copied in the head of member files
    """
    dump_info(name, info)
    for mem, f in enumerate(member_files):
        # update header if necessary
        if header_list is not None and len(header_list) > 0:
            f.header.update(header_list[mem])
        # find path
        path_name = pathlib.Path(name)
        if path_name.suffix == ".dat":
            target = path_name
        else:
            target = path_name / f"{path_name.stem}_{mem:04d}.dat"
        target.parent.mkdir(exist_ok=True)
        f.write(target)
