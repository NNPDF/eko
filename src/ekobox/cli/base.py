"""Base command definition."""

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def command():
    """Manage EKO assets and launch calculations."""
