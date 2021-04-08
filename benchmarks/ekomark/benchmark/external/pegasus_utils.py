# -*- coding: utf-8 -*-
"""
Pegasus interface
"""
import numpy as np

from eko import basis_rotation as br

def apply_pdf( tabs, pdf, labels, target_xgrid, q20):

    """ apply pdf to pegasus grid """

    # create pdfs
    pdfs = np.zeros((len(labels), len(target_xgrid)))
    for j, pid in enumerate(labels):
        if not pdf.hasFlavor(pid):
            continue
        pdfs[j] = np.array(
            [
                pdf.xfxQ2(pid, x, q20) / x
                for x in target_xgrid
            ]
        )

    # build output
    out_grid = {}
    for q2, elem in tabs.items():
        pdf_final = np.einsum("bk,bk", elem, pdfs)
        out_grid[q2] = {
            dict(zip(labels, pdf_final)),
        }
    return out_grid

def compute_pegasus_data(theory, operators, pdf, skip_pdfs, rotate_to_evolution_basis=False):
    """
    Run Pegasus to compute operators.

    Parameters
    ----------
        theory : dict
            theory card
        operators : dict
            operators card
        pdf : lhapdf_type
            pdf
        skip_pdfs : list
            list of pdfs (pid or name) to skip
        rotate_to_evolution_basis: bool
            rotate to evolution basis

    Returns
    -------
        ref : dict
            output containing: target_xgrid, values
    """
    import pegasus

    target_xgrid = operators["interpolation_xgrid"]

    # init pegasus
    nf = theory["NfFF"]
    if theory["ModEv"] == "EXA":
        imodev = 1
    elif theory["ModEv"] in ["EXP",  'decompose-expanded', 'perturbative-expanded']:
        imodev = 2
    elif theory["ModEv"] in ["TRN", "ordered-truncated"]:
        imodev = 3

    if theory["FNS"] == "FFNS":
        ivfns = 0
    else:
        ivfns = 1
    pegasus.initevol( imodev, theory["PTO"], ivfns, nf, theory["XIR"])

    #as0 = theory["alphas"]
    q20 = theory["Qref"] ** 2
    L = np.log(theory["XIR"])
    #mc2 = ( theory["kcThr"] * theory["mc"] ) ** 2
    #mb2 = ( theory["kbThr"] * theory["mb"] ) ** 2
    #mt2 = ( theory["ktThr"] * theory["mt"] ) ** 2
    pegasus.initinp(
        ivfns,
        nf,
        L,
        theory["alphas"],
        q20,
        ( theory["kcThr"] * theory["mc"] ) ** 2,
        ( theory["kbThr"] * theory["mb"] ) ** 2,
        ( theory["ktThr"] * theory["mt"] ) ** 2
    )

    if rotate_to_evolution_basis:
        ipstd = 0
        labels = list(br.evol_basis)
    else:
        ipstd = 1
        labels = list(br.flavor_basis_pids)

    for n in skip_pdfs:
        if n in labels:
            labels.remove(n)

    # compute ekos with pegaus
    out_tabs = {}
    for q2 in operators["Q2grid"]:

        me =[]
        for x in target_xgrid:
            xf, _ = pegasus.xparton( x, q2, - nf, nf, ipstd, ivfns, nf, L)
            me.append( xf )
            # temp = dict(zip(labels, xf))
            # for pid in labels:
            #     if pid not in tab.keys():
            #         tab[pid] = []
            #     tab[pid].append( temp[pid] )

        out_tabs[q2] = me

    ref = {
        "target_xgrid": target_xgrid,
        "values": apply_pdf( out_tabs, pdf, labels, target_xgrid, q20),
    }

    return ref
