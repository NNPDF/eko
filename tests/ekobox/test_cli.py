from click.testing import CliRunner

from ekobox.cli import command


def test_run():
    runner = CliRunner()
    result = runner.invoke(command, ["runcards", "-h"])
    assert "Manage EKO runcards." in result.output
