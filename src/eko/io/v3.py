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
    raw["data_version"] = 3
    return raw


def update_theory(raw: dict) -> dict:
    return raw


def update_operator(raw_op: dict, raw_th) -> dict:
    return raw_op
