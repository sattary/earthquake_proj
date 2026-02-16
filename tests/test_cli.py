from typer.testing import CliRunner
import pytest
from src.pipelines.cli import app

runner = CliRunner()


def test_cli_help():
    # Verify the app starts and help works for all subcommands
    commands = ["train", "tune", "audit", "plot", "results-suite"]
    for cmd in commands:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()


def test_cli_audit_help():
    result = runner.invoke(app, ["audit", "--help"])
    assert result.exit_code == 0
    assert "Perform Exploratory Data Analysis" in result.output


def test_cli_plot_help():
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0
    assert "Generate physical maps" in result.output
