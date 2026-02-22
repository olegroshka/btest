from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


@dataclass(frozen=True)
class LogStreamConfig:
    poll_interval_s: float = 0.15
    heartbeat_interval_s: float = 5.0
    max_bytes_per_event: int = 32_000


def _sse_event(*, data: str, event: str | None = None) -> str:
    """Format a single SSE event payload."""

    # SSE: each line is prefixed with `data:`; messages end with a blank line.
    # We split on lines to remain spec-compliant.
    lines = (data or "").splitlines() or [""]
    out = []
    if event:
        out.append(f"event: {event}")
    for ln in lines:
        out.append(f"data: {ln}")
    out.append("")
    return "\n".join(out) + "\n"


def iter_sse_log_tail(
    *,
    log_path: Path,
    is_terminal: Callable[[], bool],
    cfg: LogStreamConfig | None = None,
) -> Iterator[bytes]:
    """Tail a log file and yield SSE events as bytes.

    - Sends incremental appended text as `data:` events.
    - Emits heartbeat comments to keep proxies from buffering.
    - Terminates once `is_terminal()` is true and no new data arrives.

    The caller is responsible for setting response media_type to `text/event-stream`.
    """

    cfg = cfg or LogStreamConfig()

    # Waiting for file to appear is normal (run just started).
    last_heartbeat = 0.0
    pos = 0

    while True:
        now = time.time()
        if now - last_heartbeat >= cfg.heartbeat_interval_s:
            last_heartbeat = now
            # Comment heartbeat (SSE allows ':' lines) to avoid extra UI noise.
            yield b": heartbeat\n\n"

        try:
            if log_path.exists() and log_path.is_file():
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read(cfg.max_bytes_per_event)
                    pos = f.tell()
                if chunk:
                    yield _sse_event(data=chunk).encode("utf-8")
                    continue
        except Exception as exc:
            # Stream the error and terminate.
            yield _sse_event(data=f"[log_stream] error: {exc}", event="error").encode("utf-8")
            return

        if is_terminal():
            # One final attempt to flush anything that got written just before terminal.
            try:
                if log_path.exists() and log_path.is_file():
                    with log_path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        chunk = f.read(cfg.max_bytes_per_event)
                        pos = f.tell()
                    if chunk:
                        yield _sse_event(data=chunk).encode("utf-8")
                        continue
            except Exception:
                pass
            yield _sse_event(data="[log_stream] done", event="done").encode("utf-8")
            return

        time.sleep(cfg.poll_interval_s)
