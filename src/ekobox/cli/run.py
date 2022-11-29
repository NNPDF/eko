import click

from ._base import command


@command.command("run")
@click.argument("q2")
def subcommand(q2):
    """Launch EKO computation.

    Parameters
    ----------
    q2: sequence[float]
        sequnce of q2 to compute

    """
    print(f"Running EKO for {q2}")
