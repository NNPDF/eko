"""Legacy interface to files created with v0.14.

Although the data version v2 was never assigned to v0.14 we use it for
exactly that API version.
"""

from .paths import InternalPaths


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
    raw["data_version"] = 2
    raw["xgrid"] = raw["bases"]["xgrid"]
    del raw["bases"]
    return raw


def update_theory(raw: dict) -> dict:
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

    return raw


def update_operator(raw_op: dict, raw_th) -> dict:
    raw_op["init"] = (raw_op["mu0"], raw_th["heavy"]["num_flavs_init"])
    raw_op["mugrid"] = raw_op["mugrid"]
    del raw_op["mu0"]
    return raw_op
