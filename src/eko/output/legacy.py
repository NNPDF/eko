# -*- coding: utf-8 -*-
import io
import pathlib
import tarfile
import tempfile

import lz4.frame
import numpy as np
import yaml

from .. import version


def get_raw(obj, binarize=True, skip_q2_grid=False):
    """
    Serialize result as dict/YAML.

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
    # prepare output dict
    out = {"Q2grid": {}, "eko_version": version.__version__}
    # dump raw elements
    for f in [
        "interpolation_polynomial_degree",
        "interpolation_is_log",
        "q2_ref",
    ]:
        out[f] = obj[f]

    # list() work both for tuple and list
    out["inputpids"] = list(obj["inputpids"])
    out["targetpids"] = list(obj["targetpids"])
    # make raw lists
    # TODO: is interpolation_xgrid really needed in the output?
    for k in ["interpolation_xgrid", "targetgrid", "inputgrid"]:
        out[k] = obj[k].tolist()
    # make operators raw
    if not skip_q2_grid:
        for q2, op in obj["Q2grid"].items():
            out["Q2grid"][q2] = {}
            for k, v in op.items():
                if k == "alphas":
                    out["Q2grid"][q2][k] = float(v)
                    continue
                if binarize:
                    out["Q2grid"][q2][k] = lz4.frame.compress(v.tobytes())
                else:
                    out["Q2grid"][q2][k] = v.tolist()
    else:
        out["Q2grid"] = obj["Q2grid"]
    return out


def dump_yaml(obj, stream=None, binarize=True, skip_q2_grid=False):
    """
    Serialize result as YAML.

    Parameters
    ----------
        stream : None or stream
            if given, dump is written on it
        binarize : bool
            dump in binary format (instead of list format)
        skip_q2_grid : bool
            avoid dumping Q2grid (i.e. the actual operators) into the yaml
            file (default: ``False``)

    Returns
    -------
        dump : any
            result of dump(output, stream), i.e. a string, if no stream is given or
            Null, if written successfully to stream
    """
    # TODO explicitly silence yaml
    out = obj.get_raw(binarize, skip_q2_grid=skip_q2_grid)
    return yaml.dump(out, stream)


def dump_yaml_to_file(obj, filename, binarize=True, skip_q2_grid=False):
    """
    Writes YAML representation to a file.

    Parameters
    ----------
        filename : str
            target file name
        binarize : bool
            dump in binary format (instead of list format)
        skip_q2_grid : bool
            avoid dumping Q2grid (i.e. the actual operators) into the yaml
            file (default: ``False``)

    Returns
    -------
        ret : any
            result of dump(output, stream), i.e. Null if written successfully
    """
    with open(filename, "w", encoding="utf-8") as f:
        ret = obj.dump_yaml(f, binarize, skip_q2_grid=skip_q2_grid)
    return ret


def dump_tar(obj, tarname):
    """
    Writes representation into a tar archive containing:

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

        cls = obj.__class__
        metadata = cls(**{str(k): v for k, v in obj.items() if k != "Q2grid"})
        metadata["Q2grid"] = list(obj["Q2grid"].keys())

        yamlname = tmpdir / "metadata.yaml"
        metadata.dump_yaml_to_file(yamlname, skip_q2_grid=True)

        for kind in next(iter(obj["Q2grid"].values())).keys():
            operator = np.stack([q2[kind] for q2 in obj["Q2grid"].values()])
            stream = io.BytesIO()
            np.save(stream, operator)
            stream.seek(0)
            with lz4.frame.open((tmpdir / kind).with_suffix(".npy.lz4"), "wb") as fo:
                fo.write(stream.read())

        with tarfile.open(tarpath, "w") as tar:
            tar.add(tmpdir, arcname=tarpath.stem)


def load_yaml(stream, skip_q2_grid=False):
    """
    Load YAML representation from stream

    Parameters
    ----------
        stream : any
            source stream
        skip_q2_grid : bool
            avoid loading Q2grid (i.e. the actual operators) from the yaml
            file (default: ``False``)

    Returns
    -------
        obj : output
            loaded object
    """
    obj = yaml.safe_load(stream)
    len_tpids = len(obj["targetpids"])
    len_ipids = len(obj["inputpids"])
    len_tgrid = len(obj["targetgrid"])
    len_igrid = len(obj["inputgrid"])
    # cast lists to numpy
    for k in ["interpolation_xgrid", "inputgrid", "targetgrid"]:
        obj[k] = np.array(obj[k])
    # make operators numpy
    if not skip_q2_grid:
        for op in obj["Q2grid"].values():
            for k, v in op.items():
                if k == "alphas":
                    v = float(v)
                elif isinstance(v, list):
                    v = np.array(v)
                elif isinstance(v, bytes):
                    v = np.frombuffer(lz4.frame.decompress(v))
                    v = v.reshape(len_tpids, len_tgrid, len_ipids, len_igrid)
                op[k] = v
    return cls(obj)


def load_yaml_from_file(filename, skip_q2_grid=False):
    """
    Load YAML representation from file

    Parameters
    ----------
        filename : str
            source file name
        skip_q2_grid : bool
            avoid loading Q2grid (i.e. the actual operators) from the yaml
            file (default: ``False``)

    Returns
    -------
        obj : output
            loaded object
    """
    obj = None
    with open(filename, encoding="utf-8") as o:
        obj = load_yaml(o, skip_q2_grid)
    return obj


def load_tar(tarname):
    """
    Load tar representation from file (compliant with :meth:`dump_tar`
    output).

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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        with tarfile.open(tarpath, "r") as tar:
            tar.extractall(tmpdir)

        # metadata = cls(**{str(k): v for k, v in obj.items() if k != "Q2grid"})
        # metadata["Q2grid"] = list(obj["Q2grid"].keys())

        innerdir = list(tmpdir.glob("*"))[0]
        yamlname = innerdir / "metadata.yaml"
        metadata = load_yaml_from_file(yamlname, skip_q2_grid=True)

        grids = {}
        for fp in innerdir.glob("*.npy.lz4"):
            with lz4.frame.open(fp, "rb") as fd:
                stream = io.BytesIO(fd.read())
                stream.seek(0)
                grids[pathlib.Path(fp.stem).stem] = np.load(stream)

            fp.unlink()

        q2grid = metadata["Q2grid"]
        operator_grid = {}
        for q2, slices in zip(q2grid, zip(*grids.values())):
            operator_grid[q2] = dict(zip(grids.keys(), slices))
        metadata["Q2grid"] = operator_grid

    return metadata
