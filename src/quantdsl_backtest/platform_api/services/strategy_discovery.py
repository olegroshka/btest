from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from ..models.strategy import StrategyInfo


_SLUG_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def default_strategies_dir() -> Path:
    """Default local-first strategies directory (gitignored)."""

    return Path.cwd() / "strategies"


def examples_dir() -> Path:
    """Committed example strategies (source of bootstrap)."""

    return Path(__file__).resolve().parents[2] / "examples"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slugify(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return "strategy"
    v = _SLUG_RE.sub("_", v)
    v = v.strip("_")
    return v or "strategy"


def bootstrap_strategies(*, target_dir: Path, src_examples_dir: Path | None = None) -> bool:
    """Copy committed examples into the local strategies dir if empty.

    Returns True if bootstrap performed.
    """

    src = src_examples_dir or examples_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    # If there are any .py files already, do nothing.
    if any(p.is_file() and p.suffix == ".py" for p in target_dir.iterdir()):
        return False

    if not src.exists():
        return False

    copied = False
    for p in sorted(src.glob("*.py")):
        if not p.is_file():
            continue
        shutil.copyfile(str(p), str(target_dir / p.name))
        copied = True

    return copied


def _parse_docstring_header(source: str) -> tuple[str | None, str | None]:
    """Best-effort parse of a module docstring for (name, description)."""

    s = source.lstrip()
    if not (s.startswith('"""') or s.startswith("'''")):
        return None, None

    quote = '"""' if s.startswith('"""') else "'''"
    end = s.find(quote, len(quote))
    if end < 0:
        return None, None

    body = s[len(quote) : end].strip("\r\n ")
    if not body:
        return None, None

    lines = [ln.strip() for ln in body.splitlines()]
    title = next((ln for ln in lines if ln), None)
    rest = "\n".join([ln for ln in lines[1:] if ln]).strip() or None
    return title, rest


def discover_strategies(*, strategies_dir: Path) -> list[StrategyInfo]:
    """Discover strategy files under a directory.

    Current MVP rules:
      - Only `*.py` files, non-recursive.
      - Strategy id = filename stem.
      - We don't import/execute the module here.
    """

    out: list[StrategyInfo] = []
    if not strategies_dir.exists():
        return out

    for p in sorted(strategies_dir.glob("*.py")):
        if not p.is_file():
            continue
        sid = p.stem
        try:
            source = p.read_text(encoding="utf-8")
        except Exception:
            # Skip unreadable file
            continue

        name, desc = _parse_docstring_header(source)
        out.append(
            StrategyInfo(
                id=sid,
                path=str(p.resolve()),
                strategy_hash=sha256_text(source),
                name=name,
                description=desc,
            )
        )

    return out


def read_strategy_source(*, strategies_dir: Path, strategy_id: str) -> str:
    p = (strategies_dir / f"{strategy_id}.py").resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Strategy not found: {strategy_id}")
    return p.read_text(encoding="utf-8")


def write_strategy_source(*, strategies_dir: Path, strategy_id: str, source: str) -> Path:
    strategies_dir.mkdir(parents=True, exist_ok=True)
    sid = slugify(strategy_id)
    p = (strategies_dir / f"{sid}.py").resolve()
    p.write_text(source, encoding="utf-8")
    return p
