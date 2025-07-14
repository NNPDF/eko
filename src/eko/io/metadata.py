"""Define `eko.EKO` metadata."""

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import yaml
from packaging.version import parse

from .. import version as vmod
from ..interpolation import XGrid
from . import v1, v2
from .dictlike import DictLike
from .paths import InternalPaths
from .types import EvolutionPoint as EPoint

logger = logging.getLogger(__name__)


@dataclass
class Metadata(DictLike):
    """Manage metadata, and keep them synced on disk.

    It is possible to have a metadata view, in which the path is not actually
    connected (i.e. it is set to ``None``). In this case, no update will be
    possible, of course.

    Note
    ----
    Unfortunately, for nested structures it is not possible to detect a change
    in their attributes, so a call to :meth:`update` has to be performed
    manually.
    """

    origin: EPoint
    """Inital scale."""
    xgrid: XGrid
    """Interpolation grid."""
    # tagging information
    _path: Optional[pathlib.Path] = None
    """Path to the open dir."""
    version: str = vmod.__version__
    """Library version used to create the corresponding file."""
    data_version: int = vmod.__data_version__
    """Specs version, to which the file adheres."""

    @classmethod
    def load(cls, path: os.PathLike):
        """Load metadata from open folder.

        Parameters
        ----------
        path: os.PathLike
            the path to the open EKO folder

        Returns
        -------
        Metadata
            loaded metadata
        """
        path = pathlib.Path(path)
        paths = InternalPaths(path)
        # read raw file first to catch version
        raw = yaml.safe_load(paths.metadata.read_text(encoding="utf-8"))
        version = parse(raw["version"])
        data_version = int(raw["data_version"])
        # patch if necessary
        if data_version == 1:
            if version.major == 0 and version.minor == 13:
                raw = v1.update_metadata(paths, raw)
            elif version.major == 0 and version.minor == 14:
                raw = v2.update_metadata(paths, raw)

        # now we are ready
        content = cls.from_dict(raw)
        content._path = path
        return content

    def update(self):
        """Update the disk copy of metadata."""
        if self._path is None:
            logger.info("Impossible to set metadata, no file attached.")
        else:
            with open(InternalPaths(self._path).metadata, "w", encoding="utf8") as fd:
                yaml.safe_dump(self.raw, fd)

    @property
    def path(self):
        """Access temporary dir path.

        Raises
        ------
        RuntimeError
            if path has not been initialized before
        """
        if self._path is None:
            raise RuntimeError(
                "Access to EKO directory attempted, but not dir has been set."
            )
        return self._path

    @path.setter
    def path(self, value: pathlib.Path):
        """Set temporary dir path."""
        self._path = value

    @property
    def raw(self):
        """Override default :meth:`DictLike.raw` representation to exclude
        path."""
        return self.public_raw
