"""Launch EKO calculations."""

import pathlib
from typing import Sequence

import click
import yaml

import eko
from eko.io.runcards import OperatorCard, TheoryCard

from .base import command
from .library import OPERATOR, OUTPUT, THEORY


@command.command("run")
@click.argument("paths", nargs=-1, type=click.Path(path_type=pathlib.Path))
def subcommand(paths: Sequence[pathlib.Path]):
    r"""Launch calculation specified by PATHS.

    Compute the operator given runcards. How they are specified depends on the
    amount of arguments:

        1. only one argument provided: it is used as the folder where runcards
        and output are located (with default names)

        2. two arguments: they are used as theory and operator card paths, and
        the output is placed in the same folder of the operator card

        3. three arguments: same as two, but the third argument is used as
        output path


    Default names are:

    - theory card: "theory.yaml"
    - operator card: "operator.yaml"
    - output: "eko.tar".
    """
    if len(paths) == 1:
        theory = paths[0] / THEORY
        operator = paths[0] / OPERATOR
    elif len(paths) in [2, 3]:
        theory = paths[0]
        operator = paths[1]
    else:
        raise click.UsageError(
            f"Only one to three arguments allowed, '{len(paths)}' passed."
        )

    if len(paths) == 3:
        output = paths[2]
    else:
        output = operator.parent / OUTPUT

    tc = yaml.safe_load(theory.read_text(encoding="utf-8"))
    if "order" in tc:
        tc = TheoryCard.from_dict(tc)
    oc = yaml.safe_load(operator.read_text(encoding="utf-8"))
    if "configs" in oc:
        oc = OperatorCard.from_dict(oc)

    eko.solve(tc, oc, path=output)
