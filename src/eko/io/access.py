"""Manage file system resources access."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import exceptions


@dataclass
class AccessConfigs:
    """Configurations specified during opening of an EKO."""

    path: Path
    """The path to the permanent object."""
    readonly: bool
    "Read-only flag"
    open: bool
    "EKO status"

    def assert_open(self):
        """Assert operator is open.

        Raises
        ------
        exceptions.ClosedOperator
            if operator is closed

        """
        if not self.open:
            raise exceptions.ClosedOperator

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
            raise exceptions.ReadOnlyOperator(msg)

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
