from __future__ import annotations

import threading
import time

import pytest


@pytest.mark.unit
def test_iter_sse_log_tail_emits_done(tmp_path):
    from quantdsl_backtest.platform_api.services.log_streamer import LogStreamConfig, iter_sse_log_tail

    log_path = tmp_path / "logs.txt"
    log_path.write_text("hello\n", encoding="utf-8")

    terminal = {"v": False}

    def is_terminal() -> bool:
        return bool(terminal["v"])

    it = iter_sse_log_tail(log_path=log_path, is_terminal=is_terminal, cfg=LogStreamConfig(poll_interval_s=0.01, heartbeat_interval_s=0.01))

    out = b""

    def _consume():
        nonlocal out
        # read a few events
        for _ in range(10):
            out += next(it)

    th = threading.Thread(target=_consume, daemon=True)
    th.start()

    time.sleep(0.03)
    log_path.write_text("hello\nworld\n", encoding="utf-8")

    time.sleep(0.03)
    terminal["v"] = True

    th.join(timeout=2)
    assert b"data: world" in out
    assert b"event: done" in out

