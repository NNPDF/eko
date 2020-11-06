# -*- coding: utf-8 -*-
"""
    This file contains the output management
"""
import logging

import yaml
import numpy as np

from . import interpolation
from . import basis_rotation as br

logger = logging.getLogger(__name__)


class Output(dict):
    """
    Wrapper for the output to help with application
    to PDFs and dumping to file.
    """

    def apply_pdf(self, lhapdf_like, targetgrid=None, rotate_to_evolution_basis=False):
        """
        Apply all available operators to the input PDFs.

        Parameters
        ----------
            lhapdf_like : object
                object that provides an xfxQ2 callable (as `lhapdf <https://lhapdf.hepforge.org/>`_
                and :class:`ekomark.toyLH.toyPDF` do) (and thus is in flavor basis)
            targetgrid : list
                if given, interpolates to the pdfs given at targetgrid (instead of xgrid)

        Returns
        -------
            out_grid : dict
                output PDFs and their associated errors for the computed Q2grid
        """
        # create pdfs
        pdfs = np.zeros((len(self["pids"]), len(self["interpolation_xgrid"])))
        for j, pid in enumerate(self["pids"]):
            if not lhapdf_like.hasFlavor(pid):
                continue
            pdfs[j] = np.array(
                [
                    lhapdf_like.xfxQ2(pid, x, self["q2_ref"]) / x
                    for x in self["interpolation_xgrid"]
                ]
            )

        # build output
        out_grid = {}
        for q2, elem in self["Q2grid"].items():
            pdf_final = np.einsum("ajbk,bk", elem["operators"], pdfs)
            error_final = np.einsum("ajbk,bk", elem["operator_errors"], pdfs)
            out_grid[q2] = {
                "pdfs": dict(zip(self["pids"], pdf_final)),
                "errors": dict(zip(self["pids"], error_final)),
            }

        # rotate to evolution basis
        if rotate_to_evolution_basis:
            for q2, op in out_grid.items():
                pdf = br.rotate_flavor_to_evolution @ np.array(
                    [op["pdfs"][pid] for pid in br.flavor_basis_pids]
                )
                errors = br.rotate_flavor_to_evolution @ np.array(
                    [op["errors"][pid] for pid in br.flavor_basis_pids]
                )
                out_grid[q2]["pdfs"] = dict(zip(br.evol_basis, pdf))
                out_grid[q2]["errors"] = dict(zip(br.evol_basis, errors))

        # rotate/interpolate to target grid
        if targetgrid is not None:
            b = interpolation.InterpolatorDispatcher.from_dict(self, False)
            rot = b.get_interpolation(targetgrid)
            for q2 in out_grid:
                for pdf_label in out_grid[q2]["pdfs"]:
                    out_grid[q2]["pdfs"][pdf_label] = np.matmul(
                        rot, out_grid[q2]["pdfs"][pdf_label]
                    )
                    out_grid[q2]["errors"][pdf_label] = np.matmul(
                        rot, out_grid[q2]["errors"][pdf_label]
                    )

        return out_grid

    def get_raw(self, binarize=True):
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
        out = {
            "Q2grid": {},
        }
        # dump raw elements
        for f in [
            "interpolation_polynomial_degree",
            "interpolation_is_log",
            "q2_ref",
        ]:
            out[f] = self[f]
        out["pids"] = list(self["pids"])
        # make raw lists
        for k in ["interpolation_xgrid"]:
            out[k] = self[k].tolist()
        # make operators raw
        for q2, op in self["Q2grid"].items():
            out["Q2grid"][q2] = dict()
            for k, v in op.items():
                if binarize:
                    out["Q2grid"][q2][k] = v.tobytes()
                else:
                    out["Q2grid"][q2][k] = v.tolist()
        return out

    def dump_yaml(self, stream=None, binarize=True):
        """
        Serialize result as YAML.

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
                Null, if self is written sucessfully to stream
        """
        # TODO explicitly silence yaml
        out = self.get_raw(binarize)
        return yaml.dump(out, stream)

    def dump_yaml_to_file(self, filename, binarize=True):
        """
        Writes YAML representation to a file.

        Parameters
        ----------
            filename : string
                target file name
            binarize : bool
                dump in binary format (instead of list format)

        Returns
        -------
            ret : any
                result of dump(output, stream), i.e. Null if written sucessfully
        """
        with open(filename, "w") as f:
            ret = self.dump_yaml(f, binarize)
        return ret

    @classmethod
    def load_yaml(cls, stream):
        """
        Load YAML representation from stream

        Parameters
        ----------
            stream : any
                source stream

        Returns
        -------
            obj : output
                loaded object
        """
        obj = yaml.safe_load(stream)
        len_pids = len(obj["pids"])
        len_xgrid = len(obj["interpolation_xgrid"])
        # make list numpy
        for k in ["interpolation_xgrid"]:
            obj[k] = np.array(obj[k])
        # make operators numpy
        for op in obj["Q2grid"].values():
            for k, v in op.items():
                if isinstance(v, list):
                    v = np.array(v)
                elif isinstance(v, bytes):
                    v = np.frombuffer(v)
                    v = v.reshape(len_pids, len_xgrid, len_pids, len_xgrid)
                op[k] = v
        return cls(obj)

    @classmethod
    def load_yaml_from_file(cls, filename):
        """
        Load YAML representation from file

        Parameters
        ----------
            filename : string
                source file name

        Returns
        -------
            obj : output
                loaded object
        """
        obj = None
        with open(filename) as o:
            obj = Output.load_yaml(o)
        return obj
