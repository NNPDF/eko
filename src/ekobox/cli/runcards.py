"""Subcommand to manage runcards."""
import logging
import pathlib

from ..cards import generate_operator as opgen
from ..cards import generate_theory as thgen
from . import library as lib
from .base import command

_logger = logging.getLogger(__name__)

DESTINATION = pathlib.Path.cwd().absolute() / "runcards"
"""Default destination for generated files"""

option_dest = lib.destination(DESTINATION)


@command.group("runcards")
def subcommand():
    """Manage EKO runcards."""


@subcommand.command("example")
@option_dest
def sub_example(destination: pathlib.Path):
    """Generate example runcards.

    Pay attention that they are in no way intended as defaults, but just
    examples to quickstart.

    """
    destination.mkdir(parents=True, exist_ok=True)
    thgen(0, 1.65, path=destination / "theory.yaml")
    opgen([1e5], path=destination / "operator.yaml")
    _logger.info(f"Runcards generated to '{destination}'")
