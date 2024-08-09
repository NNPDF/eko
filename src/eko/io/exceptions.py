"""IO generic exceptions."""

import os


class OutputError(Exception):
    """Generic Output Error."""


class OutputExistsError(FileExistsError, OutputError):
    """Output file already exists."""


class OutputNotTar(ValueError, OutputError):
    """Specified file is not a .tar archive."""


class OperatorLoadingError(ValueError, OutputError):
    """Issue encountered while loading an operator."""


class OperatorLocationError(ValueError, OutputError):
    """Path supposed to store an operator in wrong location."""

    def __init__(self, path: os.PathLike):
        self.path = path
        super().__init__(f"Path '{path}' not in operators folder")
