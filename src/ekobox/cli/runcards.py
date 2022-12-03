"""Subcommand to manage runcards."""
import logging

from . import library as lib
from .base import command

_logger = logging.getLogger(__name__)


@command.group("runcards")
def subcommand():
    """Manage EKO runcards."""


@subcommand.command("example")
@lib.option_dest
def sub_example(destination):
    """Generate example runcards.

    Pay attention that they are in no way intended as defaults, but just
    examples to quickstart.

    """
    _logger.info("Generated")
