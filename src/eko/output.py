# -*- coding: utf-8 -*-
"""
    This file contains the output management
"""
import logging
import warnings

import yaml
import lz4.frame
import numpy as np

from . import basis_rotation as br
from . import interpolation

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
            rotate_to_evolution_basis : bool
                if True rotate to evoluton basis

        Returns
        -------
            out_grid : dict
                output PDFs and their associated errors for the computed Q2grid
        """
        if rotate_to_evolution_basis:
            return self.apply_pdf_flavor(
                lhapdf_like, targetgrid, br.rotate_flavor_to_evolution
            )
        return self.apply_pdf_flavor(lhapdf_like, targetgrid)

    def apply_pdf_flavor(self, lhapdf_like, targetgrid=None, flavor_rotation=None):
        """
        Apply all available operators to the input PDFs.

        Parameters
        ----------
            lhapdf_like : object
                object that provides an xfxQ2 callable (as `lhapdf <https://lhapdf.hepforge.org/>`_
                and :class:`ekomark.toyLH.toyPDF` do) (and thus is in flavor basis)
            targetgrid : list
                if given, interpolates to the pdfs given at targetgrid (instead of xgrid)
            flavor_rotation : np.ndarray
                Rotation matrix in flavor space

        Returns
        -------
            out_grid : dict
                output PDFs and their associated errors for the computed Q2grid
        """
        # create pdfs
        pdfs = np.zeros((len(self["inputpids"]), len(self["inputgrid"])))
        for j, pid in enumerate(self["inputpids"]):
            if not lhapdf_like.hasFlavor(pid):
                continue
            pdfs[j] = np.array(
                [
                    lhapdf_like.xfxQ2(pid, x, self["q2_ref"]) / x
                    for x in self["inputgrid"]
                ]
            )

        # build output
        out_grid = {}
        for q2, elem in self["Q2grid"].items():
            pdf_final = np.einsum("ajbk,bk", elem["operators"], pdfs)
            error_final = np.einsum("ajbk,bk", elem["operator_errors"], pdfs)
            out_grid[q2] = {
                "pdfs": dict(zip(self["targetpids"], pdf_final)),
                "errors": dict(zip(self["targetpids"], error_final)),
            }

        # rotate to evolution basis
        if flavor_rotation is not None:
            for q2, op in out_grid.items():
                pdf = flavor_rotation @ np.array(
                    [op["pdfs"][pid] for pid in br.flavor_basis_pids]
                )
                errors = flavor_rotation @ np.array(
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

    def xgrid_reshape(self, targetgrid=None, inputgrid=None):
        """
        Changes the operators to have in the output targetgrid and/or in the input inputgrid.

        The operation is inplace.

        Parameters
        ----------
            targetgrid : None or list
                xgrid for the target
            inputgrid : None or list
                xgrid for the input
        """
        # calling with no arguments is an error
        if targetgrid is None and inputgrid is None:
            raise ValueError("Nor inputgrid nor targetgrid was given")
        # now check to the current status
        if (
            targetgrid is not None
            and len(targetgrid) == len(self["targetgrid"])
            and np.allclose(targetgrid, self["targetgrid"])
        ):
            targetgrid = None
            warnings.warn("The new targetgrid is close to the current targetgrid")
        if (
            inputgrid is not None
            and len(inputgrid) == len(self["inputgrid"])
            and np.allclose(inputgrid, self["inputgrid"])
        ):
            inputgrid = None
            warnings.warn("The new inputgrid is close to the current inputgrid")
        # after the checks: if there is still nothing to do, skip
        if targetgrid is None and inputgrid is None:
            logger.debug("Nothing done.")
            return

        # construct matrices
        if targetgrid is not None:
            b = interpolation.InterpolatorDispatcher(
                self["targetgrid"],
                self["interpolation_polynomial_degree"],
                self["interpolation_is_log"],
                False,
            )
            target_rot = b.get_interpolation(targetgrid)
            self["targetgrid"] = targetgrid
        if inputgrid is not None:
            b = interpolation.InterpolatorDispatcher(
                inputgrid,
                self["interpolation_polynomial_degree"],
                self["interpolation_is_log"],
                False,
            )
            input_rot = b.get_interpolation(self["inputgrid"])
            self["inputgrid"] = inputgrid

        # build new grid
        out_grid = {}
        for q2, elem in self["Q2grid"].items():
            ops = elem["operators"]
            errs = elem["operator_errors"]
            if targetgrid is not None and inputgrid is None:
                ops = np.einsum("ij,ajbk->aibk", target_rot, ops)
                errs = np.einsum("ij,ajbk->aibk", target_rot, errs)
            elif inputgrid is not None and targetgrid is None:
                ops = np.einsum("ajbk,kl->ajbl", ops, input_rot)
                errs = np.einsum("ajbk,kl->ajbl", errs, input_rot)
            else:
                ops = np.einsum("ij,ajbk,kl->aibl", target_rot, ops, input_rot)
                errs = np.einsum("ij,ajbk,kl->aibl", target_rot, errs, input_rot)
            out_grid[q2] = dict(operators=ops, operator_errors=errs)
        # set in self
        self["Q2grid"] = out_grid

    def flavor_reshape(self, targetbasis=None, inputbasis=None):
        """
        Changes the operators to have in the output targetbasis and/or in the input inputbasis.

        The operation is inplace.

        Parameters
        ----------
            targetbasis : numpy.ndarray
                target rotation specified in the flavor basis
            inputbasis : None or list
                input rotation specified in the flavor basis
        """
        # calling with no arguments is an error
        if targetbasis is None and inputbasis is None:
            raise ValueError("Nor inputbasis nor targetbasis was given")
        # now check to the current status
        if targetbasis is not None and np.allclose(
            targetbasis, np.eye(len(self["targetpids"]))
        ):
            targetbasis = None
            warnings.warn("The new targetbasis is close to current basis")
        if inputbasis is not None and np.allclose(
            inputbasis, np.eye(len(self["inputpids"]))
        ):
            inputbasis = None
            warnings.warn("The new inputbasis is close to current basis")
        # after the checks: if there is still nothing to do, skip
        if targetbasis is None and inputbasis is None:
            logger.debug("Nothing done.")
            return

        # flip input around
        if inputbasis is not None:
            inv_inputbasis = np.linalg.inv(inputbasis)

        # build new grid
        out_grid = {}
        for q2, elem in self["Q2grid"].items():
            ops = elem["operators"]
            errs = elem["operator_errors"]
            if targetbasis is not None and inputbasis is None:
                ops = np.einsum("ca,ajbk->cjbk", targetbasis, ops)
                errs = np.einsum("ca,ajbk->cjbk", targetbasis, errs)
            elif inputbasis is not None and targetbasis is None:
                ops = np.einsum("ajbk,bd->ajdk", ops, inv_inputbasis)
                errs = np.einsum("ajbk,bd->ajdk", errs, inv_inputbasis)
            else:
                ops = np.einsum("ca,ajbk,bd->cjdk", targetbasis, ops, inv_inputbasis)
                errs = np.einsum("ca,ajbk,bd->cjdk", targetbasis, errs, inv_inputbasis)
            out_grid[q2] = dict(operators=ops, operator_errors=errs)
        # set in self
        self["Q2grid"] = out_grid
        # drop PIDs
        if inputbasis is not None:
            self["inputpids"] = np.full(len(self["inputpids"]), np.nan)
        if targetbasis is not None:
            self["targetpids"] = np.full(len(self["targetpids"]), np.nan)

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
        out["inputpids"] = list(self["inputpids"])
        out["targetpids"] = list(self["targetpids"])
        # make raw lists
        # TODO: is interpolation_xgrid really needed in the output?
        for k in ["interpolation_xgrid", "targetgrid", "inputgrid"]:
            out[k] = self[k].tolist()
        # make operators raw
        for q2, op in self["Q2grid"].items():
            out["Q2grid"][q2] = dict()
            for k, v in op.items():
                if binarize:
                    out["Q2grid"][q2][k] = lz4.frame.compress(v.tobytes())
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
        len_tpids = len(obj["targetpids"])
        len_ipids = len(obj["inputpids"])
        len_tgrid = len(obj["targetgrid"])
        len_igrid = len(obj["inputgrid"])
        # cast lists to numpy
        for k in ["interpolation_xgrid", "inputgrid", "targetgrid"]:
            obj[k] = np.array(obj[k])
        # make operators numpy
        for op in obj["Q2grid"].values():
            for k, v in op.items():
                if isinstance(v, list):
                    v = np.array(v)
                elif isinstance(v, bytes):
                    v = np.frombuffer(lz4.frame.decompress(v))
                    v = v.reshape(len_tpids, len_tgrid, len_ipids, len_igrid)
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
