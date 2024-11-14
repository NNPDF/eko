"""Inspect EKO content."""

import pathlib

import click
import rich

from eko.io import EKO

from .base import command
from .library import OUTPUT


@command.group("inspect")
@click.option(
    "-p",
    "--path",
    type=click.Path(path_type=pathlib.Path, exists=True),
    default=pathlib.Path.cwd() / OUTPUT,
)
@click.pass_context
def subcommand(ctx, path: pathlib.Path):
    """Inspect EKO content."""
    ctx.obj = ctx.with_resource(EKO.read(path))


@subcommand.command("mu2grid")
@click.pass_obj
def sub_mu2(operator: EKO):
    """Check operator's mu2grid."""
    rich.print_json(data=operator.mu2grid)


@subcommand.command("cards")
@click.pass_obj
def sub_cards(operator: EKO):
    """Check operator's mu2grid."""
    try:
        theory_card = operator.theory_card.raw
    except KeyError as e:
        theory_card = dict(error=f"Key '{e.args}' missing in theory_card")

    try:
        operator_card = operator.operator_card.raw
    except KeyError as e:
        operator_card = dict(error=f"Key '{e.args}' missing in operator_card")

    rich.print_json(data=dict(theory=theory_card, operator=operator_card))
