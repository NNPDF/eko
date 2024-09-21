"""Utilities to manipulate unstructured IO.

The content is treated independently on the particular data content, but
as generic uknown data in an abstract file format, e.g. a tar archive or
YAML data file, as opposed to structured YAML representing a specific
runcard.
"""

import os
from pathlib import Path
from tarfile import TarFile, TarInfo
from typing import Optional, Sequence


def is_within_directory(directory: os.PathLike, target: os.PathLike) -> bool:
    """Check if target path is contained in directory.

    Thanks to TrellixVulnTeam for the `idea
    <https://github.com/NNPDF/eko/pull/154>`_.

    Parameters
    ----------
    directory:
        the directory where the target is supposed to be contained
    target:
        the target file to check
    """
    abs_dir = Path(directory).absolute()
    abs_target = Path(target).absolute()

    return abs_dir == abs_target or abs_dir in abs_target.parents


def safe_extractall(
    tar: TarFile,
    path: Optional[os.PathLike] = None,
    members: Optional[Sequence[TarInfo]] = None,
    *,
    numeric_owner: bool = False,
):
    """Extract a tar archive avoiding CVE-2007-4559 issue.

    Thanks to TrellixVulnTeam for the `contribution
    <https://github.com/NNPDF/eko/pull/154>`_.

    All undocumented parameters have the same meaning of the analogue ones in
    :meth:`TarFile.extractall`.

    Parameters
    ----------
    tar:
        the tar archive object to be extracted
    path:
        the path to extract to, if not specified the current directory is used
    """
    if path is None:
        path = Path.cwd()
    path = Path(path)

    for member in tar.getmembers():
        member_path = path / member.name
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)
