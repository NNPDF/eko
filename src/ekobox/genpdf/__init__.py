# -*- coding: utf-8 -*-
import copy
import pathlib
import shutil

import numpy as np
from banana import toy

from eko import basis_rotation as br

from . import export, flavors, load


def take_data(parent_pdf_set=None, members=False, xgrid=None, Q2grid=None):
    """
    Auxiliary function for `generate_pdf`.

    It provides the info, the heads of the member files and the blocks
    to be generated to `generate_pdf`.

    Parameters
    ----------
        parent_pdf_set : None or str or dict
            the PDF set to be used as parent set
        members : bool
            if true every member of the parent is loaded
        xgrid : list(float)
            produced x grid if given
        Q2grid : list(float)
            produced Q2 grid if given

    Returns
    -------
        info : dict
            info dictionary
        heads : list(str)
            heads of member files if necessary
        blocks : list(dict)
            data blocks
    """
    if xgrid is None:
        xgrid = np.geomspace(1e-9, 1, 240)
    if Q2grid is None:
        Q2grid = np.geomspace(1.3, 1e5, 35)
    # collect blocks
    all_blocks = []
    info = None
    heads = []
    if parent_pdf_set is None:
        parent_pdf_set = {
            pid: lambda x, _Q2: x * (1 - x) for pid in br.flavor_basis_pids
        }
    if isinstance(parent_pdf_set, str):
        if parent_pdf_set in ["toylh", "toy"]:
            info = copy.deepcopy(load.Toy_info)
            toylh = toy.mkPDF("", 0)
            all_blocks.append(
                [generate_block(toylh.xfxQ2, xgrid, Q2grid, br.flavor_basis_pids)]
            )
        else:
            info = load.load_info_from_file(parent_pdf_set)
            # iterate on members
            for m in range(int(info["NumMembers"])):
                head, blocks = load.load_blocks_from_file(parent_pdf_set, m)
                heads.append(head)
                all_blocks.append(blocks)
                if not members:
                    break
    elif isinstance(parent_pdf_set, dict):
        info = copy.deepcopy(load.template_info)
        all_blocks.append(
            [
                generate_block(
                    lambda pid, x, Q2: 0.0
                    if pid not in parent_pdf_set
                    else parent_pdf_set[pid](x, Q2),
                    xgrid,
                    Q2grid,
                    br.flavor_basis_pids,
                )
            ]
        )
    else:
        raise ValueError("Unknown parent pdf type")
    return info, heads, all_blocks


def generate_pdf(
    name,
    labels,
    parent_pdf_set=None,
    members=False,
    info_update=None,
    install=False,
    xgrid=None,
    Q2grid=None,
):
    """
    Generate a new PDF from a parent PDF with a set of flavors.

    If `parent_pdf_set` is the name of an available PDF set,
    it will be used as parent. In order to use the toy PDF
    as parent, it is enough to set `parent_pdf_set` to "toy" or "toylh".
    If `parent_pdf_set` is not specified, a debug PDF constructed as
    x * (1-x) for every flavor will be used as parent.
    It is also possible to provide custom functions for each flavor
    in the form of a dictionary: `{pid: f(x,Q2)}`.

    With `labels` it is possible to pass a list of PIDs or evolution basis
    combinations to keep in the generated PDF. In order to project
    on custom combinations of PIDs, it is also possible to pass a list
    containing the desired factors for each flavor.

    The default behaviour is to generate only one member for a PDF set
    (the zero member) but it can be changed setting to True the `members` flag.

    The `info_update` argument is a dictionary and provide to the user as a way
    to change the info file of the generated PDF set. If a key of `info_update`
    matches with one key of the standard info file, the information
    are updated, otherwise they are simply added.

    Turning True the value of the `install` flag, it is possible to automatically
    install the generated PDF to the lhapdf directory. By default install is False.

    Parameters
    ----------
        name : str
            target name
        labels :
            list of flavors
        parent_pdf_set :
            parent PDF
        all : bool
            iterate on members
        install : bool
            install on LHAPDF path
        xgrid : list(float)
            produced x grid if given
        Q2grid : list(float)
            produced Q2 grid if given

    Examples
    --------
        To generate a PDF with a fixed function `f(x,Q2)` for some flavors
        you can use the following snippet:

        >>> # f = lambda x,Q2 ... put the desired function here
        >>> # mask = [list of active PIDs]
        >>> generate_pdf(name, labels, parent_pdf_set={pid: f for pid in mask})

        The |API| also provides the possibility to extract arbitrary flavor combinations:
        using the debug PDF settings we can construct a "anti-QED-singlet" combination that
        is usefull in debugging DIS codes since it does not couple in |LO|, but only
        through the pure-singlet contributions (starting at |NNLO|)

        >>> from eko import basis_rotation as br
        >>> from ekobox import genpdf
        >>> import numpy as np
        >>> anti_qed_singlet = np.zeros_like(br.flavor_basis_pids, dtype=np.float_)
        >>> anti_qed_singlet[br.flavor_basis_pids.index(1)] = -4
        >>> anti_qed_singlet[br.flavor_basis_pids.index(-1)] = -4
        >>> anti_qed_singlet[br.flavor_basis_pids.index(2)] = 1
        >>> anti_qed_singlet[br.flavor_basis_pids.index(-2)] = 1
        >>> genpdf.generate_pdf("anti_qed_singlet", [anti_qed_singlet])
    """
    pathlib.Path(name).mkdir(exist_ok=True)
    # Checking label basis
    is_evol = False
    flavor_combinations = labels
    if flavors.is_evolution_labels(labels):
        is_evol = True
        flavor_combinations = flavors.evol_to_flavor(labels)
    elif flavors.is_pid_labels(labels):
        labels = np.array(labels, dtype=np.int_)
        flavor_combinations = flavors.pid_to_flavor(labels)

    # labels = verify_labels(args.labels)
    info, heads, all_blocks = take_data(
        parent_pdf_set, members, xgrid=xgrid, Q2grid=Q2grid
    )

    # filter the PDF
    new_all_blocks = []
    for b in all_blocks:
        new_all_blocks.append(flavors.project(b, flavor_combinations))

    # changing info file according to user choice
    if info_update is not None:
        info.update(info_update)
    # write
    info["Flavors"] = [int(pid) for pid in br.flavor_basis_pids]
    info["NumFlavors"] = len(br.flavor_basis_pids)
    if is_evol:
        info["ForcePositive"] = 0
    info["NumMembers"] = len(new_all_blocks)
    # exporting
    export.dump_set(name, info, new_all_blocks, pdf_type_list=heads)

    # install
    if install:
        install_pdf(name)


def install_pdf(name):
    """
    Install set into LHAPDF.

    The set to be installed has to be in the current directory.

    Parameters
    ----------
        name : str
            source pdf name
    """
    import lhapdf  # pylint: disable=import-error, import-outside-toplevel

    print(f"install_pdf {name}")
    target = pathlib.Path(lhapdf.paths()[0])
    src = pathlib.Path(name)
    # shutil.move only accepts paths since 3.9 so we need to cast
    # https://docs.python.org/3/library/shutil.html?highlight=shutil#shutil.move
    shutil.move(str(src), str(target))


def generate_block(xfxQ2, xgrid, Q2grid, pids):
    """
    Generate an LHAPDF data block from a callable

    Parameters
    ----------
        xfxQ2 : callable
            LHAPDF like callable
        Q2grid : list(float)
            Q grid
        pids : list(int)
            Flavors list
        xgrid : list(float)
            x grid

    Returns
    -------
        dict
            PDF block
    """
    block = dict(Q2grid=Q2grid, pids=pids, xgrid=xgrid)
    data = []
    for x in xgrid:
        for Q2 in Q2grid:
            data.append(np.array([xfxQ2(pid, x, Q2) for pid in pids]))
    block["data"] = np.array(data)
    return block
