from __future__ import annotations

import queue
import threading
import time

import pytest


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_health(*, port: int, timeout_s: float = 8.0) -> None:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health")
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.05)
    raise AssertionError("Server did not start")


def _start_server_in_thread(*, port: int) -> None:
    def _run():
        import importlib
        import uvicorn

        # IMPORTANT: reload app module so monkeypatches (default dirs/db path) take effect
        # even when running as part of a larger test suite.
        from quantdsl_backtest.platform_api import main as platform_main

        importlib.reload(platform_main)
        app = platform_main.app

        uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="warning")

    th = threading.Thread(target=_run, daemon=True)
    th.start()


def _read_sse_events(*, url: str, timeout_s: float = 5.0) -> list[str]:
    """Read raw SSE lines until we see an 'event: done' or timeout."""

    import httpx

    events: list[str] = []
    t0 = time.time()

    with httpx.stream("GET", url, timeout=timeout_s) as r:
        assert r.status_code == 200
        buf: list[str] = []
        for line in r.iter_lines():
            if time.time() - t0 > timeout_s:
                break
            # httpx yields '' on blank line
            if line is None:
                continue
            if line == "":
                if buf:
                    events.append("\n".join(buf))
                    if any(l.startswith("event: done") for l in buf):
                        return events
                    buf = []
                continue
            buf.append(line)

    return events


@pytest.mark.unit
def test_run_logs_stream_endpoint_flushes_existing_logs(tmp_path, monkeypatch):
    """Integration-style unit test: SSE endpoint yields existing content and terminates for completed runs."""

    # Hermetic db + strategies dir MUST be patched before app import/server start.
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )
    monkeypatch.setenv("QDSL_PLATFORM_RUNNER_PROCESS_POOL", "1")

    (strategies_dir / "s_sse_done.py").write_text(
        """
from quantdsl_backtest.dsl.strategy import Strategy


def build_strategy() -> Strategy:
    raise RuntimeError('intentional test failure: s_sse_done')
""".lstrip(),
        encoding="utf-8",
    )

    port = _free_port()
    _start_server_in_thread(port=port)
    _wait_health(port=port)

    import httpx

    resp = httpx.post(f"http://127.0.0.1:{port}/api/runs", json={"strategy_id": "s_sse_done", "params": {}})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    # Wait until terminal
    t0 = time.time()
    while time.time() - t0 < 20:
        r2 = httpx.get(f"http://127.0.0.1:{port}/api/runs/{run_id}")
        assert r2.status_code == 200
        if r2.json()["run"]["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    # Ensure logs exist via non-stream endpoint
    logs = httpx.get(f"http://127.0.0.1:{port}/api/runs/{run_id}/logs")
    assert logs.status_code == 200, logs.text
    assert "intentional test failure" in str(logs.json().get("logs") or "")

    # Now SSE stream should immediately emit that content then event: done.
    events = _read_sse_events(url=f"http://127.0.0.1:{port}/api/runs/{run_id}/logs/stream", timeout_s=6.0)
    joined = "\n\n".join(events)
    assert "intentional test failure" in joined
    assert "event: done" in joined


@pytest.mark.unit
def test_run_logs_stream_endpoint_tails_while_running(tmp_path, monkeypatch):
    """SSE endpoint should stream incremental lines while a run is still running."""

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )
    monkeypatch.setenv("QDSL_PLATFORM_RUNNER_PROCESS_POOL", "1")

    # Strategy prints a few lines with sleeps to force streaming.
    (strategies_dir / "s_sse_tail.py").write_text(
        """
import time
from quantdsl_backtest.dsl.strategy import Strategy


def build_strategy() -> Strategy:
    print('line_1', flush=True)
    time.sleep(0.4)
    print('line_2', flush=True)
    time.sleep(0.4)
    raise RuntimeError('boom')
""".lstrip(),
        encoding="utf-8",
    )

    port = _free_port()
    _start_server_in_thread(port=port)
    _wait_health(port=port)

    import httpx

    resp = httpx.post(f"http://127.0.0.1:{port}/api/runs", json={"strategy_id": "s_sse_tail", "params": {}})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    # Start SSE reader in background so we can observe partial stream.
    q: queue.Queue[list[str]] = queue.Queue()

    def _reader():
        evs = _read_sse_events(url=f"http://127.0.0.1:{port}/api/runs/{run_id}/logs/stream", timeout_s=10.0)
        q.put(evs)

    th = threading.Thread(target=_reader, daemon=True)
    th.start()

    # Bound: wait for completion
    t0 = time.time()
    while time.time() - t0 < 30:
        r2 = httpx.get(f"http://127.0.0.1:{port}/api/runs/{run_id}")
        assert r2.status_code == 200
        if r2.json()["run"]["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    events = q.get(timeout=15)
    joined = "\n\n".join(events)

    assert "line_1" in joined
    assert "line_2" in joined
    assert "event: done" in joined
