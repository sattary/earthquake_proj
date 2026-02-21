from typer.testing import CliRunner
from src.pipelines.cli import app

runner = CliRunner()


def test_cli_help():
    # Verify the app starts and help works for all subcommands
    commands = ["train", "tune", "plot-history", "plot-all-local", "results-suite"]
    for cmd in commands:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()


def test_cli_plot_all_local_help():
    result = runner.invoke(app, ["plot-all-local", "--help"])
    assert result.exit_code == 0
    assert "Master command" in result.output
