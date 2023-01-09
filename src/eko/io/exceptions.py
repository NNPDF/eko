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


class ReadOnlyOperator(RuntimeError, OutputError):
    """It is not possible to write on a read-only operator.

    In particular, the behavior would be deceitful, since writing is possible
    in-memory and even on the temporary folder.
    But eventually, no writing will happen on a persistent archive, so any
    modification is lost after exiting the program.

    """


class ClosedOperator(RuntimeError, OutputError):
    """It is not possible to write on nor to read from a closed operator.

    This is milder issue than :class:`ReadOnlyOperator`, since in this case not
    even the writing on the temporary folder would be possible.

    Instead, it will look like you can access some properties, but the operator
    is actually closed, so it should not be used any longer in general.
    However, for extremely simple properties, like those available in memory
    from :class:`eko.io.struct.Metadata` or :class:`eko.io.struct.AccessConfigs`, there
    is no need to raise on read, since those properties are actually available,
    but they should always raise on writing, since there is no persistence for
    the content written, and it can be deceitful.

    Still, the level of protection will be mild, since a thoruough protection
    would clutter a lot the code, requiring a lot of maintenance.
    "We are adult here".

    """
