"""Subcommand to manage runcards."""

import logging
import pathlib

import numpy as np

from .. import cards
from . import library as lib
from .base import command

_logger = logging.getLogger(__name__)

DESTINATION = pathlib.Path.cwd().absolute() / "runcards"
"""Default destination for generated files."""

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
    theory = cards.example.theory()
    theory.order = (1, 0)
    cards.dump(theory.raw, path=destination / "theory.yaml")
    operator = cards.example.operator()
    operator.init = (1.65, 4)
    operator.mugrid = [(np.sqrt(1e5), 5)]
    cards.dump(operator.raw, path=destination / "operator.yaml")
    _logger.info(f"Runcards generated to '{destination}'")
