"""Library of reusable options and elements."""
import logging
import pathlib

import click

DESTINATION = pathlib.Path.cwd().absolute() / "theory"
"""Default destination for generated files"""

option_dest = click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=DESTINATION,
    help="Alternative destination path to store the resulting table (default: $PWD/theory)",
)
