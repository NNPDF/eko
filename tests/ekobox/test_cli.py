# -*- coding: utf-8 -*-

from click.testing import CliRunner

from ekobox.cli import command


def test_run():
    runner = CliRunner()
    result = runner.invoke(command, ["run", "a"])
    assert "Running EKO for" in result.output
