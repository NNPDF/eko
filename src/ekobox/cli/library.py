"""Library of reusable options and elements."""

import pathlib

import click

THEORY = "theory.yaml"
OPERATOR = "operator.yaml"
OUTPUT = "eko.tar"


def destination(dest: pathlib.Path):
    """Build click option for destination.

    Parameters
    ----------
    dest : pathlib.Path
        default destination

    Returns
    -------
    click.option
        generated option
    """
    return click.option(
        "-d",
        "--destination",
        type=click.Path(path_type=pathlib.Path, exists=True),
        default=dest,
        help="Alternative destination path to store the resulting table (default: $PWD/theory)",
    )
