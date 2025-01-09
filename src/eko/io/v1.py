"""Legacy interface to files created with v0.13.

Although the data version v1 was already before v0.13 we only support
that API version.
"""

from .paths import InternalPaths
import numpy as np
import yaml

def update_metadata(paths: InternalPaths, raw: dict) -> dict:
    """Modify the raw metadata to the new format.

    Parameters
    ----------
    paths:
        base paths to the EKO
    raw:
        raw yaml content

    Returns
    -------
    dict
        compatible raw yaml content
    """
    raw["data_version"] = 1
    raw["xgrid"] = raw["rotations"]["xgrid"]
    del raw["rotations"]
    raw_theory = yaml.safe_load(paths.theory_card.read_text(encoding="utf-8"))
    raw["origin"] = (np.sqrt(raw["mu20"]), raw_theory["heavy"]["num_flavs_init"])
    del raw["mu20"]
    return raw


def update_theory(raw: dict) -> dict:
    """Modify the raw theory card to the new format.

    Parameters
    ----------
    raw:
        raw yaml content

    Returns
    -------
    dict
        compatible raw yaml content
    """
    raw["couplings"]["ref"] = (
        raw["couplings"]["scale"],
        raw["couplings"]["num_flavs_ref"],
    )
    # adjust couplings
    for key in ["num_flavs_ref", "max_num_flavs", "scale"]:
        del raw["couplings"][key]
    # adjust heavy
    for key in ["intrinsic_flavors", "num_flavs_init", "num_flavs_max_pdf"]:
        del raw["heavy"][key]

    # update old names
    if "use_fhmv" in raw:
        raw["use_fhmruvv"] = raw["use_fhmv"]
        del raw["use_fhmv"]

    return raw


def update_operator(raw_op: dict, raw_th) -> dict:
    """Modify the raw operator card to the new format.

    Parameters
    ----------
    raw_op:
        raw operator yaml content
    raw_th:
        raw theory yaml content
    Returns
    -------
    dict
        compatible raw yaml content
    """
    raw_op["init"] = (raw_op["mu0"], raw_th["heavy"]["num_flavs_init"])
    del raw_op["mu0"]
    return raw_op
