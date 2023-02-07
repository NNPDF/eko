"""Upgrade old files."""
import pathlib
from typing import Optional

import click

from eko.io import legacy

from .base import command


@command.command("convert")
@click.argument("old", type=click.Path(path_type=pathlib.Path, exists=True))
@click.option("-n", "--new", type=click.Path(path_type=pathlib.Path), default=None)
def subcommand(old: pathlib.Path, new: Optional[pathlib.Path]):
    """Convert old EKO files to new format.

    The OLD file path is used also for the new one, appending "-new" to its
    stem, unless it is explicitly specified with the dedicated option.
    """
    if new is None:
        new = old.parent / old.with_stem(old.stem + "-new")

    legacy.load_tar(old, new)
