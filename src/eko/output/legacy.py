# -*- coding: utf-8 -*-
"""Support legacy storage formats."""
import copy
import dataclasses
import io
import os
import pathlib
import tarfile
import tempfile
from typing import Optional, TextIO, Union

import lz4.frame
import numpy as np
import yaml

from .. import version
from . import struct


def get_raw(eko: struct.EKO, binarize: bool = True):
    """Serialize result as dict/YAML.

    This maps the original numpy matrices to lists.

    Parameters
    ----------
    binarize : bool
        dump in binary format (instead of list format)

    Returns
    -------
    out : dict
        dictionary which will be written on output

    """
    obj = eko.raw

    # prepare output dict
    out = {"Q2grid": {}, "eko_version": version.__version__}
    out["Q0"] = obj["Q0"]
    # dump raw elements
    for sec in ["configs", "rotations"]:
        for key, value in obj[sec].items():
            if key.startswith("_"):
                key = key[1:]
            if "grid" in key and value is not None:
                value = value["grid"]
            out[key] = value

    out["interpolation_xgrid"] = out["xgrid"]
    del out["xgrid"]

    # make operators raw
    for q2, op in eko.items():
        q2 = float(q2)
        out["Q2grid"][q2] = {}
        if op is not None:
            for k, v in dataclasses.asdict(op).items():
                if binarize:
                    out["Q2grid"][q2][k] = lz4.frame.compress(v.tobytes())
                else:
                    out["Q2grid"][q2][k] = v.tolist()

    return out


def tocard(raw: dict) -> dict:
    """Upgrade raw representation to new card.

    Parameters
    ----------
    raw: dict
        legacy raw representation of Output

    Returns
    -------
    dict
        new format operator card

    """
    card = copy.deepcopy(raw)

    card["rotations"] = {}
    card["rotations"]["xgrid"] = raw["interpolation_xgrid"]
    card["rotations"]["pids"] = raw["pids"]
    for basis in ("inputgrid", "targetgrid", "inputpids", "targetpids"):
        card["rotations"][basis] = raw[basis]

    card["configs"] = {}
    for field in dataclasses.fields(struct.Configs):
        card["configs"][field.name] = raw[field.name]
        del card[field.name]

    return card


def dump_yaml(
    obj: struct.EKO,
    stream: Optional[TextIO] = None,
    binarize: bool = True,
):
    """Serialize result as YAML.

    Parameters
    ----------
    stream : None or stream
        if given, dump is written on it
    binarize : bool
        dump in binary format (instead of list format)

    Returns
    -------
    dump : any
        result of dump(output, stream), i.e. a string, if no stream is given or
        Null, if written successfully to stream

    """
    out = get_raw(obj, binarize)
    return yaml.dump(out, stream)


def dump_yaml_to_file(
    obj: struct.EKO,
    filename: Union[str, os.PathLike],
    binarize: bool = True,
):
    """Write YAML representation to a file.

    Parameters
    ----------
    filename : str
        target file name
    binarize : bool
        dump in binary format (instead of list format)

    Returns
    -------
    ret : any
        result of dump(output, stream), i.e. Null if written successfully

    """
    with open(filename, "w", encoding="utf-8") as f:
        ret = dump_yaml(obj, f, binarize)
    return ret


def dump_tar(obj: struct.EKO, tarname: Union[str, os.PathLike]):
    """Write representation into a tar archive.

    The written archive will contain:

    - metadata (in YAML)
    - operator (in numpy ``.npy`` format)

    Parameters
    ----------
    tarname : str
        target file name

    """
    tarpath = pathlib.Path(tarname)
    if tarpath.suffix != ".tar":
        raise ValueError(f"'{tarname}' is not a valid tar filename, wrong suffix")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        metadata = {str(k): v for k, v in get_raw(obj).items() if k != "Q2grid"}
        metadata["Q2grid"] = obj.Q2grid.tolist()

        yamlname = tmpdir / "metadata.yaml"
        with open(yamlname, "w", encoding="utf-8") as fd:
            yaml.dump(metadata, fd)

        for kind in ["operator", "error"]:
            elements = []
            for q2, op in obj.items():
                el = getattr(op, kind)
                elements.append(el)
            operator = np.stack(elements)
            stream = io.BytesIO()
            np.save(stream, operator)
            stream.seek(0)
            with lz4.frame.open((tmpdir / kind).with_suffix(".npy.lz4"), "wb") as fo:
                fo.write(stream.read())

        with tarfile.open(tarpath, "w") as tar:
            tar.add(tmpdir, arcname=tarpath.stem)


def load_yaml(stream: TextIO) -> struct.EKO:
    """Load YAML representation from stream.

    Parameters
    ----------
    stream : TextIO
        source stream

    Returns
    -------
    obj : output
        loaded object

    """
    obj = tocard(yaml.safe_load(stream))
    len_tpids = len(obj["rotations"]["targetpids"])
    len_ipids = len(obj["rotations"]["inputpids"])
    len_tgrid = len(obj["rotations"]["targetgrid"])
    len_igrid = len(obj["rotations"]["inputgrid"])
    # make operators numpy
    for op in obj["Q2grid"].values():
        for k, v in op.items():
            if isinstance(v, list):
                v = np.array(v)
            elif isinstance(v, bytes):
                v = np.frombuffer(lz4.frame.decompress(v))
                v = v.reshape(len_tpids, len_tgrid, len_ipids, len_igrid)
            op[k] = v

    return struct.EKO.new(theory={}, operator=obj)


def load_yaml_from_file(filename: Union[str, os.PathLike]) -> struct.EKO:
    """Load YAML representation from file.

    Parameters
    ----------
    filename : str
        source file name

    Returns
    -------
    obj : output
        loaded object

    """
    obj = None
    with open(filename, encoding="utf-8") as o:
        obj = load_yaml(o)
    return obj


def load_tar(tarname: Union[str, os.PathLike]) -> struct.EKO:
    """Load tar representation from file.

    Compliant with :meth:`dump_tar` output.

    Parameters
    ----------
    tarname : str
        source tar name

    Returns
    -------
    obj : output
        loaded object

    """
    tarpath = pathlib.Path(tarname)

    operator_grid = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        with tarfile.open(tarpath, "r") as tar:
            tar.extractall(tmpdir)

        # load metadata
        innerdir = list(tmpdir.glob("*"))[0]
        yamlname = innerdir / "metadata.yaml"
        with open(yamlname, encoding="utf-8") as fd:
            metadata = tocard(yaml.safe_load(fd))

        # get actual grids
        grids = {}
        for fp in innerdir.glob("*.npy.lz4"):
            with lz4.frame.open(fp, "rb") as fd:
                stream = io.BytesIO(fd.read())
                stream.seek(0)
                grids[pathlib.Path(fp.stem).stem] = np.load(stream)

        q2grid = metadata["Q2grid"]
        for q2, slices in zip(q2grid, zip(*grids.values())):
            operator_grid[q2] = dict(zip(grids.keys(), slices))

    # now eveything is in place
    eko = struct.EKO.new(theory={}, operator=metadata)
    for q2, op in operator_grid.items():
        eko[q2] = struct.Operator.from_dict(op)

    return eko
