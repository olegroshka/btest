from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _default_cache_root() -> Path:
    # Repo convention: local_cache/ is the LMDB directory used by ArcticDB.
    # We keep this local and deterministic.
    return Path(__file__).resolve().parents[1] / "local_cache"


def _rm_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reset the local ArcticDB cache directory used by quantdsl_backtest. "
            "Useful when the LMDB store becomes corrupted (e.g. MDB_INVALID)."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=_default_cache_root(),
        help="Path to the cache root directory (default: ./local_cache)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Move cache directory to <path>.bak instead of deleting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without modifying anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not prompt for confirmation.",
    )

    args = parser.parse_args()
    cache_path: Path = args.path

    action = "move" if args.backup else "delete"

    # Always print a minimal banner so it's clear the script ran
    print(f"Arctic cache reset tool")
    print(f"  target: {cache_path}")
    print(f"  exists: {cache_path.exists()}")
    print(f"  action: {action}{' (dry-run)' if args.dry_run else ''}")

    if not cache_path.exists():
        return 0

    if args.dry_run:
        if args.backup:
            print(f"[dry-run] Would move {cache_path} -> {cache_path}.bak")
        else:
            print(f"[dry-run] Would delete {cache_path}")
        return 0

    if not args.force:
        print(f"This will {action}: {cache_path}")
        resp = input("Proceed? [y/N]: ").strip().lower()
        if resp not in {"y", "yes"}:
            print("Aborted.")
            return 1

    if args.backup:
        backup_path = cache_path.with_suffix(cache_path.suffix + ".bak")
        if cache_path.suffix == "":
            backup_path = Path(str(cache_path) + ".bak")

        if backup_path.exists():
            i = 1
            while True:
                candidate = Path(str(backup_path) + f".{i}")
                if not candidate.exists():
                    backup_path = candidate
                    break
                i += 1

        print(f"Moving {cache_path} -> {backup_path}")
        shutil.move(str(cache_path), str(backup_path))
        return 0

    print(f"Deleting {cache_path}")
    _rm_tree(cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
