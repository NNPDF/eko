"""Manage file system resources access."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import exceptions


class ReadOnlyOperator(RuntimeError, exceptions.OutputError):
    """It is not possible to write on a read-only operator.

    In particular, the behavior would be deceitful, since writing is
    possible in-memory and even on the temporary folder. But eventually,
    no writing will happen on a persistent archive, so any modification
    is lost after exiting the program.
    """


class ClosedOperator(RuntimeError, exceptions.OutputError):
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


@dataclass
class AccessConfigs:
    """Configurations specified during opening of an EKO."""

    path: Optional[Path]
    """The path to the permanent object."""
    readonly: bool
    "Read-only flag"
    open: bool
    "EKO status"

    @property
    def read(self):
        """Check reading permission.

        Reading access is always granted on open operator.
        """
        return self.open

    @property
    def write(self):
        """Check writing permission."""
        return self.open and not self.readonly

    def assert_open(self):
        """Assert operator is open.

        Raises
        ------
        exceptions.ClosedOperator
            if operator is closed
        """
        if not self.open:
            raise ClosedOperator

    def assert_writeable(self, msg: Optional[str] = None):
        """Assert operator is writeable.

        Raises
        ------
        exceptions.ClosedOperator
            see :meth:`assert_open`
        exceptions.ReadOnlyOperator
            if operators has been declared read-only
        """
        if msg is None:
            msg = ""

        self.assert_open()
        if self.readonly:
            raise ReadOnlyOperator(msg)
