# -*- coding: utf-8 -*-

from click.testing import CliRunner

from ekobox.genpdf.cli import cli


def test_genpdf_CLI_messages():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate"])
    assert "Error: Missing argument 'NAME'." in result.output
    result = runner.invoke(cli, ["install"])
    assert "Error: Missing argument 'NAME'." in result.output
    result = runner.invoke(cli, ["generate", "TestEmptyLabels"])
    assert result.exception and "Labels must contain at least one element" in str(
        result.exception
    )
    result = runner.invoke(cli, ["generate", "--help"])
    assert "-p, --parent-pdf-set TEXT" in result.output
    assert "-m, --members" in result.output
    assert "-i, --install" in result.output
