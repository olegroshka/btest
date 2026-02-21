from __future__ import annotations

"""Unified test runner for this repo.

Goals:
- One command to run the full automated suite by default (unit + slow, excluding manual).
- Clear, copy/pasteable command echoing.
- Reliable output capture: stream to console AND write to a log file.
- Opt-in smoke/manual suites that require external setup.

Typical usage (PowerShell):
  uv run python scripts/run_tests.py                 # default: unit + slow (not manual/smoke)
  uv run python scripts/run_tests.py --all           # REALLY all: unit + slow + platform + smoke
  uv run python scripts/run_tests.py --unit          # unit only
  uv run python scripts/run_tests.py --slow          # slow only (not manual/smoke)
  uv run python scripts/run_tests.py --platform      # platform subset
  uv run python scripts/run_tests.py --smoke         # smoke (manual) + server preflight warning
  uv run python scripts/run_tests.py --manual --slow # include manual tests in slow suite

Forward extra args to pytest after "--":
  uv run python scripts/run_tests.py -- --maxfail=1 -x -k signal

Notes:
- We prefer running pytest via "uv run pytest" when uv is available, but we fall back to
  "python -m pytest" if not.
"""

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _quote_arg(a: str) -> str:
    # Pretty printing only. This is not used for execution.
    if a == "":
        return '""'
    if any(c in a for c in " \t\n\"'"):
        return '"' + a.replace('"', '\\"') + '"'
    return a


def _format_cmd(argv: list[str]) -> str:
    return " ".join(_quote_arg(a) for a in argv)


def _pytest_launcher() -> list[str]:
    # Prefer uv (matches README), but keep a graceful fallback.
    if shutil.which("uv"):
        return ["uv", "run", "pytest"]
    return [sys.executable, "-m", "pytest"]


def _default_log_dir() -> Path:
    return _REPO_ROOT / ".test_logs"


def _read_platform_port(default: int = 8000) -> int:
    # Mirror tests_slow/smoke behavior: read .platform_ui/server.port if present.
    try:
        p = (_REPO_ROOT / ".platform_ui" / "server.port")
        if p.exists():
            return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return default
    return default


def _tcp_health_check(host: str, port: int, timeout_s: float = 0.8) -> bool:
    # Avoid extra deps. Just attempt a TCP connect; the smoke tests do the HTTP check.
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def _warn_if_server_down() -> None:
    port = _read_platform_port(default=8000)
    host = "127.0.0.1"

    # Prefer a real /health HTTP check (matches the smoke tests) so we don't print
    # a misleading warning when the server is already up.
    try:
        import httpx  # type: ignore

        url = f"http://{host}:{port}/health"
        r = httpx.get(url, timeout=1.5)
        if int(r.status_code) == 200:
            return
    except Exception:
        # fall back to a quick TCP check
        if _tcp_health_check(host, port):
            return

    # Option A: do NOT auto-start. Just warn with exact commands.
    print("\n[SMOKE PRECHECK] Platform server does not appear to be running.")
    print(f"[SMOKE PRECHECK] Expected listening on http://{host}:{port}/ (or at least TCP {host}:{port}).")
    print("[SMOKE PRECHECK] Start it in another terminal with one of:")
    print(f"  .\\scripts\\run_platform_ui.ps1 -Port {port}")
    print(f"  uv run python scripts/run_platform_ui.py --port {port}")
    print("[SMOKE PRECHECK] Continuing anyway (pytest will fail fast with details).\n")


def _tee_run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    log_file: Path | None,
) -> int:
    print("\n" + "=" * 88)
    print("RUN:")
    print(_format_cmd(argv))
    print("=" * 88)

    log_fh = None
    try:
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fh = log_file.open("w", encoding="utf-8", errors="replace", newline="\n")
            log_fh.write(f"# {cwd}\n")
            log_fh.write(f"# {_format_cmd(argv)}\n\n")
            log_fh.flush()

        p = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert p.stdout is not None
        for line in p.stdout:
            # Preserve pytest output as-is.
            sys.stdout.write(line)
            if log_fh is not None:
                log_fh.write(line)
        p.wait()

        rc = int(p.returncode or 0)
        # Pytest exit code 5 means "no tests collected".
        # When the runner executes multiple suites (e.g., default unit+slow) and the user
        # filters with -k / -m, it's normal for some suites to collect nothing.
        # Treat that as success.
        if rc == 5:
            rc = 0

        if log_fh is not None:
            log_fh.write(f"\n# exit_code={rc}\n")
            log_fh.flush()

        if log_file is not None:
            print(f"\n[log] saved: {log_file}")

        return rc
    finally:
        try:
            if log_fh is not None:
                log_fh.close()
        except Exception:
            pass


def _is_flag(a: str) -> bool:
    return a.startswith("-")


def _has_filtering_passthrough(passthrough: list[str]) -> bool:
    """Return True if user passed args that likely narrow collection.

    This is a heuristic used only for the *default* selection.
    """

    for a in passthrough:
        if a in {"-k", "-m", "--lf", "--ff", "--last-failed", "--failed-first"}:
            return True
        if a.startswith("-k") or a.startswith("-m"):
            return True
        if a.startswith("tests") or a.endswith(".py") or "::" in a:
            return True
        # If user passes a bare positional that isn't an option, it's likely a test selector.
        if not _is_flag(a):
            return True

    return False


def _build_suites(args: argparse.Namespace, *, passthrough: list[str]) -> list[tuple[str, list[str]]]:
    """Return list of (suite_name, pytest_args).

    Suite meanings:
    - unit: tests/unit (pytest testpaths)
    - slow: tests_slow excluding smoke (automated integration-ish tests)
    - platform: tests_slow/platform (includes API e2e + Playwright tests; some are manual/net)
    - web_ui: Playwright-only UI tests under tests_slow/platform
    - smoke: tests_slow/smoke (live server required; manual)
    - ui: all UI tests (web_ui + smoke)
    """

    # Explicit suite flags.
    any_explicit = bool(
        args.unit
        or args.slow
        or args.platform
        or args.smoke
        or args.web_ui
        or args.ui
        or args.all
    )

    if args.all:
        run_unit = True
        run_slow = True
        run_platform = True
        run_web_ui = True
        run_smoke = True
    elif args.ui:
        run_unit = False
        run_slow = False
        run_platform = False
        run_web_ui = True
        run_smoke = True
    elif not any_explicit:
        # Default: full automated suite (unit + slow).
        # If user is filtering (e.g. -k / path), default becomes unit-only to avoid confusing
        # deselections across suites.
        if _has_filtering_passthrough(passthrough):
            run_unit, run_slow = True, False
        else:
            run_unit, run_slow = True, True
        run_platform = False
        run_web_ui = False
        run_smoke = False
    else:
        run_unit = bool(args.unit)
        run_slow = bool(args.slow)
        run_platform = bool(args.platform)
        run_web_ui = bool(args.web_ui)
        run_smoke = bool(args.smoke)

    suites: list[tuple[str, list[str]]] = []

    if run_unit:
        suites.append(("unit", []))

    if run_slow:
        suites.append(("slow", ["tests_slow", "--ignore", "tests_slow/smoke"]))

    if run_platform:
        suites.append(("platform", ["tests_slow/platform"]))

    if run_web_ui:
        # Playwright-only web UI tests. These are typically marked manual.
        suites.append(("web_ui", ["tests_slow/platform", "-k", "playwright"]))

    if run_smoke:
        suites.append(("smoke", ["tests_slow/smoke"]))

    return suites


def _marker_args(args: argparse.Namespace, suite_name: str) -> list[str]:
    # Smoke suite is always manual.
    if suite_name == "smoke":
        return ["-m", "manual"]

    # Web UI tests are Playwright; they are manual-marked in this repo.
    if suite_name == "web_ui":
        return ["-m", "manual"]

    # For other suites, include manual only if requested or when --all is set.
    if args.all or args.manual:
        return ["-m", "manual"]

    return []


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified test runner (unit + slow by default)")
    parser.add_argument("--all", action="store_true", help="Run EVERYTHING: unit + slow + platform + web-ui + smoke")
    parser.add_argument("--ui", action="store_true", help="Run all UI tests: Playwright web UI + smoke UI/API (manual; server required for smoke)")
    parser.add_argument("--unit", action="store_true", help="Run fast unit tests only")
    parser.add_argument("--slow", action="store_true", help="Run slow tests only (tests_slow excluding smoke)")
    parser.add_argument("--platform", action="store_true", help="Run tests under tests_slow/platform (API e2e + UI e2e)")
    parser.add_argument("--web-ui", dest="web_ui", action="store_true", help="Run Playwright web UI tests (tests_slow/platform filtered to '*playwright*'; manual)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests under tests_slow/smoke (manual; requires live server)")
    parser.add_argument("--manual", action="store_true", help="Include manual tests in selected suite(s)")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (adds '-n auto' if pytest-xdist is installed)",
    )
    parser.add_argument(
        "--log-dir",
        default=str(_default_log_dir()),
        help="Directory to write log files into (default: .test_logs)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write log files",
    )

    known, rest = parser.parse_known_args(argv)

    # Support passthrough after '--' (preferred), but also allow parse_known_args passthrough.
    passthrough: list[str] = []
    if "--" in rest:
        idx = rest.index("--")
        passthrough = rest[idx + 1 :]
    else:
        passthrough = rest

    return known, passthrough


def _maybe_parallel_args(args: argparse.Namespace) -> list[str]:
    if not args.parallel:
        return []

    # Only attach -n auto if xdist appears importable.
    try:
        import pytest_xdist  # type: ignore  # noqa: F401

        return ["-n", "auto"]
    except Exception:
        # Don't fail. Just warn once.
        print("[warn] --parallel requested but pytest-xdist isn't available; running single-process.")
        return []


def main(argv: list[str] | None = None) -> int:
    args, passthrough = _parse_args(argv or sys.argv[1:])

    launcher = _pytest_launcher()
    log_dir = Path(args.log_dir)

    suites = _build_suites(args, passthrough=passthrough)
    if not suites:
        print("Nothing selected. Use --unit/--slow/--smoke/--platform/--all or run with no flags for default.")
        return 2

    results: list[tuple[str, int, Path | None]] = []

    for suite_name, suite_pytest_args in suites:
        if suite_name == "smoke":
            _warn_if_server_down()

        cmd: list[str] = []
        cmd += launcher
        cmd += ["-q"]
        cmd += suite_pytest_args
        cmd += _marker_args(args, suite_name)
        cmd += _maybe_parallel_args(args)
        cmd += passthrough

        log_file = None
        if not args.no_log:
            log_file = log_dir / f"{_now_stamp()}_{suite_name}.log"

        rc = _tee_run(cmd, cwd=_REPO_ROOT, env=os.environ.copy(), log_file=log_file)
        results.append((suite_name, rc, log_file))

        if rc != 0:
            print(f"\n[FAIL] suite '{suite_name}' exited with code {rc}")
            return rc

    # Compact summary (useful when the default runs multiple suites)
    print("\nSummary:")
    for suite_name, rc, log_file in results:
        st = "OK" if rc == 0 else f"FAIL({rc})"
        if log_file is None:
            print(f"  - {suite_name}: {st}")
        else:
            print(f"  - {suite_name}: {st}  (log: {log_file})")

    print("\n[OK] all selected suites passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

