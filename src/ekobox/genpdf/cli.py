"""Defines the CLI - see :doc:`here </code/genpdf>`."""

import click

from . import generate_pdf, install_pdf


@click.group()
def cli():
    """Generate and install fake PDFs.

    See online documentation for a detailed explanation and examples.
    """


@cli.command("generate")
@click.argument("name")
@click.argument("labels", nargs=-1)
@click.option("-p", "--parent-pdf-set", default=None, help="parent pdf set")
@click.option("-m", "--members", is_flag=True, help="generate all the members")
@click.option("-i", "--install", is_flag=True, help="install into LHAPDF")
def cli_generate_pdf(name, labels, parent_pdf_set, members, install):
    """Generate a new PDF from a parent set with given flavors."""
    if len(labels) == 0:
        raise ValueError("Labels must contain at least one element")
    return generate_pdf(name, labels, parent_pdf_set, members, None, install)


@cli.command("install")
@click.argument("name")
def cli_install_pdf(name):
    """Install the PDF in LHAPDF directory."""
    return install_pdf(name)
