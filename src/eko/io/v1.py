"""Legacy interface to files created with v0.13.

Although the data version v1 was already before v0.13 we only support
that API version.
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
    return raw
