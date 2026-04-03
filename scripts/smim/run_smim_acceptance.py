"""SMIM Acceptance Test runner.

Runs the full (or partial) SMIM acceptance suite and prints a structured
gate report. Exits with code 0 when all collected tests pass, 1 otherwise.

Usage:
  uv run python scripts/smim/run_smim_acceptance.py              # full suite
  uv run python scripts/smim/run_smim_acceptance.py --section graph
  uv run python scripts/smim/run_smim_acceptance.py --section spectral
  uv run python scripts/smim/run_smim_acceptance.py --section kalman
  uv run python scripts/smim/run_smim_acceptance.py --section kim
  uv run python scripts/smim/run_smim_acceptance.py --section pid
  uv run python scripts/smim/run_smim_acceptance.py --section tda
  uv run python scripts/smim/run_smim_acceptance.py --section pipeline
  uv run python scripts/smim/run_smim_acceptance.py --section benchmarks
  uv run python scripts/smim/run_smim_acceptance.py -v           # verbose
  uv run python scripts/smim/run_smim_acceptance.py -- -x        # stop on first fail

Extra pytest args can be appended after "--":
  uv run python scripts/smim/run_smim_acceptance.py -- --maxfail=3 -k "P_1 or P_2"
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ACCEPTANCE_DIR = "tests/acceptance/smim"

# Section → pytest -k filter expression
_SECTION_FILTERS: dict[str, str] = {
    "graph":       "A_GR or I_GR or D_GR or A_NE or I_NE or D_NE or A_AO or I_AO or A_SP or I_SP or I_NM or A_NM",
    "spectral":    "A_SC or I_SC or R_SC or D_SC or A_PL or I_PL or R_PL or D_PL or A_HD or I_HD or R_HD or D_HD or I_DV or R_DV or A_DMD or I_DMD or R_DMD or D_DMD",
    "mode":        "A_MDL or I_MDL or A_LZ or I_LZ or A_RG",
    "kalman":      "A_KF or I_KF or R_KF or D_KF or A_EM or I_EM or D_EM",
    "kim":         "A_KIM or I_KIM or R_KIM or D_KIM",
    "observability": "A_OB or I_OB",
    "phase":       "I_OP or A_GL or I_GL or A_CI or I_CI",
    "pid":         "A_PID or I_PID",
    "te":          "A_TE or I_TE or R_TE",
    "tda":         "A_TDA or I_TDA or R_TDA",
    "benchmarks":  "A_PB or I_PB or I_MB or A_EB",
    "pipeline":    "P_1 or P_2 or P_3 or P_4",
}


def _find_pytest() -> list[str]:
    """Return the command prefix to invoke pytest (prefer uv run pytest)."""
    if shutil.which("uv"):
        return ["uv", "run", "pytest"]
    return [sys.executable, "-m", "pytest"]


def _build_cmd(
    section: str | None,
    verbose: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = _find_pytest()
    cmd += [_ACCEPTANCE_DIR, "--tb=short"]
    if verbose:
        cmd += ["-v"]
    else:
        cmd += ["-q"]
    if section:
        section_key = section.lower()
        if section_key not in _SECTION_FILTERS:
            valid = ", ".join(sorted(_SECTION_FILTERS))
            print(f"Unknown section {section!r}. Valid: {valid}", file=sys.stderr)
            sys.exit(2)
        cmd += ["-k", _SECTION_FILTERS[section_key]]
    cmd += extra_args
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SMIM acceptance tests and print a gate report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pass extra pytest args after '--': run_smim_acceptance.py -- -x -k P_1",
    )
    parser.add_argument(
        "--section", "-s",
        metavar="NAME",
        help=(
            "Run only one section. Choices: "
            + ", ".join(sorted(_SECTION_FILTERS))
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Pass -v to pytest (show each test name).",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest (prefix with --).",
    )

    args = parser.parse_args(argv)

    # Strip leading '--' separator from extra args
    extra_args = args.extra
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd = _build_cmd(args.section, args.verbose, extra_args)

    date_str = datetime.date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"  SMIM Acceptance Suite  —  {date_str}")
    if args.section:
        print(f"  Section: {args.section}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
