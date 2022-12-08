"""Launch EKO calculations, with legacy Q2grid mode."""
import pathlib

import click
import rich

from eko.io import EKO

from .base import command
from .library import OUTPUT

pass_operator = click.make_pass_decorator(EKO)


@command.group("inspect")
@click.option(
    "-p",
    "--path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd() / OUTPUT,
)
@click.pass_context
def subcommand(ctx, path: pathlib.Path):
    """Inspect EKO content."""
    ctx.obj = EKO.load(path)


@subcommand.command("mu2grid")
@pass_operator
def sub_mu2(operator: EKO):
    """Check operator's mu2grid."""
    rich.print_json(data=operator.mu2grid.tolist())


@subcommand.command("cards")
@pass_operator
def sub_cards(operator: EKO):
    """Check operator's mu2grid."""
    rich.print_json(data=dict(theory=operator.theory, operator=operator.operator_card))
