"""Define `eko.EKO` metadata."""
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import yaml

from .. import version as vmod
from .bases import Bases
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
    bases: Bases
    """Manipulation information, describing the current status of the EKO (e.g.
    `inputgrid` and `targetgrid`).
    """
    # tagging information
    _path: Optional[pathlib.Path] = None
    """Path to temporary dir."""
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
        bool
            loaded metadata

        """
        path = pathlib.Path(path)
        content = cls.from_dict(
            yaml.safe_load(InternalPaths(path).metadata.read_text(encoding="utf-8"))
        )
        content._path = path
        return content

    def update(self):
        """Update the disk copy of metadata."""
        if self._path is None:
            logger.info("Impossible to set metadata, no file attached.")
        else:
            with open(InternalPaths(self._path).metadata, "w") as fd:
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
        """Override default :meth:`DictLike.raw` representation to exclude path."""
        return self.public_raw
