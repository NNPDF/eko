"""Launch EKO calculations, with legacy Q2grid mode."""
import click

from .base import command


@command.command("run")
@click.argument("q2")
def subcommand(q2):
    """Launch EKO computation.

    Compute the EKO to evolve to a given Q2 value.

    """
    print(f"Running EKO for {q2}")
