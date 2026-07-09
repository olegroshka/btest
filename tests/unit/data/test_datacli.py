from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("cmd2")  # datacli needs the `cli` extra (cmd2 + rich)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import datacli  # type: ignore  # noqa: E402


def test_eodhd_plugin_command_map() -> None:
    plugin = datacli.EodhdPlugin()
    # /fetch maps to the eodhd CLI `refresh` subcommand; args pass through.
    assert plugin.build_argv("fetch", ["--fast", "--run"]) == [
        "refresh",
        "--fast",
        "--run",
    ]
    assert plugin.build_argv("status", []) == ["status"]
    assert plugin.build_argv("qc", ["--lane", "us_common"]) == [
        "qc",
        "--lane",
        "us_common",
    ]
    assert plugin.build_argv("lanes", []) == ["lanes"]


def test_eodhd_plugin_command_names() -> None:
    names = datacli.EodhdPlugin().command_names()
    assert set(names) == {"status", "fetch", "qc", "lanes", "probe", "config"}


def test_sources_registry() -> None:
    assert "eodhd" in datacli.SOURCES
    assert "fred" in datacli.SOURCES
    # yahoo is still a load-only adapter (no ops tooling yet).
    assert "yahoo" not in datacli.SOURCES
    assert "yahoo" in datacli.LOAD_ONLY


def test_fred_plugin_command_names() -> None:
    assert set(datacli.FredPlugin().command_names()) == {"status", "fetch", "config"}


def test_fred_parse_fetch() -> None:
    series, start, end = datacli.FredPlugin._parse_fetch(
        ["gdp", "unrate", "--start", "2020-01-01"]
    )
    assert series == ["GDP", "UNRATE"]  # upper-cased
    assert start == "2020-01-01"
    assert end  # defaults to today
    # no series -> empty (caller shows usage)
    assert datacli.FredPlugin._parse_fetch([])[0] == []


def test_argv_parses_string_and_arg_list() -> None:
    # plain string -> shlex split
    assert datacli.DataCli._argv("--fast --run") == ["--fast", "--run"]

    class _Stmt:
        arg_list = ["--lane", "us_etf"]

    # cmd2 Statement-like object -> use its arg_list
    assert datacli.DataCli._argv(_Stmt()) == ["--lane", "us_etf"]
