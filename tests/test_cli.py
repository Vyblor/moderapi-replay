"""Tests for CLI commands."""

from typer.testing import CliRunner

from moderapi.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "moderapi-replay" in result.output.lower() or "validate" in result.output.lower()


def test_cli_estimate():
    result = runner.invoke(app, ["estimate", "We use Perspective API for TOXICITY"])
    assert result.exit_code == 0
    assert "Complexity" in result.output or "complexity" in result.output


def test_cli_replay_missing_file():
    result = runner.invoke(app, ["replay", "nonexistent.jsonl"])
    assert result.exit_code != 0
