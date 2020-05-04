# -*- coding: utf-8 -*-
"""
    This file contains the output management
"""
import logging

import numpy as np
from yaml import dump

import eko.interpolation as interpolation
from eko.evolution_operator import apply_PDF_to_operator

logger = logging.getLogger(__name__)


class Output(dict):
    """
        Wrapper for the output to help with application
        to PDFs and dumping to file
    """

    def apply_PDF(self, input_pdfs, targetgrid=None):
        """
            Apply all available operators to the input PDFs.

            Parameters
            ----------
                input_pdfs : dict
                    input PDFs as dictionary {name: PDF}
                targetgrid : array
                    if given, interpolates to the pdfs given at targetgrid (instead of xgrid)

            Returns
            ---------
                out_grid : dict
                    output PDFs and their associated errors for the computed q2_grid
        """
        # TODO rotation from the evolution basis to flavor basis? if yes, here!
        # turn input_pdfs into lists
        input_lists = {}
        for k in input_pdfs:
            l = []
            for x in self["xgrid"]:
                l.append(input_pdfs[k](x))
            input_lists[k] = np.array(l)

        # build output
        in_grid = self["q2_grid"]
        out_grid = {}
        for q2 in in_grid:
            pdfs, errs = apply_PDF_to_operator(in_grid[q2], input_lists)
            out_grid[q2] = {"pdfs": pdfs, "errors": errs}

        # interpolate to target grid
        if targetgrid is not None:
            b = interpolation.InterpolatorDispatcher(
                self["xgrid"],
                self["polynomial_degree"],
                self["is_log_interpolation"],
                False,
            )
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


    def dump_YAML(self, stream=None):
        """
            Serialize result as YAML.

            Parameters
            ----------
                stream : (Default: None)
                    if given, dump is written on it

            Returns
            -------
                dump : Any
                    result of dump(output, stream), i.e. a string, if no stream is given or
                    Null, self is written sucessfully to stream
        """
        # prepare output dict
        out = {
            "q2_grid": {},
        }
        # dump raw elements
        for f in ["polynomial_degree","is_log_interpolation","q2_ref"]:
            out[f] = self[f]
        # make raw lists
        for k in ["xgrid"]:
            out[k] = self[k].tolist()
        # make operators raw
        for q2 in self["q2_grid"]:
            out_op = {
                "operators": {},
                "operator_errors": {}
            }
            # map matrices
            for k in self["q2_grid"][q2]["operators"]:
                out_op["operators"][k] = self["q2_grid"][q2]["operators"][k].tolist()
                out_op["operator_errors"][k] = self["q2_grid"][q2]["operator_errors"][k].tolist()
            out["q2_grid"][q2] = out_op
        # everything else is up to pyyaml
        return dump(out, stream)


    def write_YAML_to_file(self, filename):
        """
            Writes YAML representation to a file.

            Parameters
            ----------
                filename : string
                    target file name

            Returns
            -------
                ret :
                    result of dump(output, stream), i.e. Null if written sucessfully
        """
        with open(filename, "w") as f:
            ret = self.dump_YAML(f)
        return ret
