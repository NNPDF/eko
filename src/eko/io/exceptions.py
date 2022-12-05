"""IO generic exceptions."""


class OutputError(Exception):
    """Generic Output Error."""


class OutputExistsError(FileExistsError, OutputError):
    """Output file already exists."""


class OutputNotTar(ValueError, OutputError):
    """Specified file is not a .tar archive."""


class OperatorLoadingError(ValueError, OutputError):
    """Issue encountered while loading an operator."""
