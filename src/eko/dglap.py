# -*- coding: utf-8 -*-
"""
    This file contains the main loop for the DGLAP calculations.
"""
import logging
import numpy as np
from yaml import dump

from eko.runner import Runner
from eko.interpolation import InterpolatorDispatcher

logger = logging.getLogger(__name__)


def run_dglap(setup):
    r"""
        This function takes a DGLAP theory configuration dictionary
        and performs the solution of the DGLAP equations.

        The EKO :math:`\hat E_{k,j}(t_1\leftarrow t_0)` is determined in order
        to fullfill the following evolution

        .. math::
            f(x_k,t_1) = \hat E_{k,j}(t_1\leftarrow t_0) f(x_j,t_0)

        Parameters
        ----------
            setup : dict
                input card - see :doc:`/Code/IO`

        Returns
        -------
            output : dict
                output dictionary - see :doc:`/Code/IO`
    """
    r = Runner(setup)
    output = r.get_output()
    return output


# TODO: move to the operator class
def apply_operator(ret, input_pdfs, targetgrid=None):
    """
        Apply all available operators to the input PDFs.

        Parameters
        ----------
            ret : dict
                operator definitions - return value of `run_dglap`
            input_pdfs : dict
                input PDFs as dictionary {name: PDF}
            targetgrid : array
                if given, interpolates to the pdfs given at targetgrid (instead of xgrid)

        Returns
        ---------
            outs : dict
                output PDFs
            out_errors : dict
                associated error to the output PDFs
    """
    # TODO rotation from the evolution basis to flavor basis? if yes, here!
    # turn input_pdfs into lists
    input_lists = {}
    for k in input_pdfs:
        l = []
        for x in ret["xgrid"]:
            l.append(input_pdfs[k](x))
        input_lists[k] = np.array(l)
    # build output
    outs = {}
    out_errors = {}
    first_ret = list(ret["q2_grid"].values())[0]
    for k in first_ret["operators"]:
        out_key, in_key = k.split(".")
        # basis vector available?
        if in_key not in input_pdfs:
            # thus can I not complete the calculation for this out_key?
            if out_key in outs:
                outs[out_key] = None
            continue
        op = first_ret["operators"][k]
        op_err = first_ret["operator_errors"][k]
        # is out_key new?
        if out_key not in outs:
            # set output
            outs[out_key] = np.matmul(op, input_lists[in_key])
            out_errors[out_key] = np.matmul(op_err, input_lists[in_key])
        else:
            # is out_key already blocked?
            if outs[out_key] is None:
                continue
            # else add to it
            outs[out_key] += np.matmul(op, input_lists[in_key])
            out_errors[out_key] += np.matmul(op_err, input_lists[in_key])

    # interpolate to target grid
    if targetgrid is not None:
        b = InterpolatorDispatcher(ret["xgrid"],ret["polynomial_degree"],ret["is_log_interpolation"],False)
        rot = b.get_interpolation(targetgrid)
        for k in outs:
            outs[k] = np.matmul(rot, outs[k])
            out_errors[k] = np.matmul(rot, out_errors[k])

    return outs, out_errors


def get_YAML(ret, stream=None):
    """
        Serialize result as YAML.

        Parameters
        ----------
            ret : dict
                DGLAP result
            stream : (Default: None)
                if given, is written on

        Returns
        -------
            dump :
                result of dump(output, stream), i.e. a string, if no stream is given or
                the Null, if output is written sucessfully to stream
    """
    out = {
        "polynomial_degree": ret["polynomial_degree"],
        "log": ret["log"],
        "operators": {},
        "operator_errors": {},
    }
    # make raw lists - we might want to do somthing more numerical here
    for k in ["xgrid"]:
        out[k] = ret[k].tolist()
    for k in ret["operators"]:
        out["operators"][k] = ret["operators"][k].tolist()
        out["operator_errors"][k] = ret["operator_errors"][k].tolist()
    return dump(out, stream)


def write_YAML_to_file(ret, filename):
    """
        Writes YAML representation to a file.

        Parameters
        ----------
            ret : dict
                DGLAP result
            filename : string
                target file name

        Returns
        -------
            ret :
                result of dump(output, stream), i.e. Null if written sucessfully
    """
    with open(filename, "w") as f:
        ret = get_YAML(ret, f)
    return ret
