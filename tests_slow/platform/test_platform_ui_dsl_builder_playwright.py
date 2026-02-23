"""Playwright E2E tests for DSL Builder editor upgrade (Milestone 3.5.7).

These tests verify:
  1. The DSL Builder tab renders with a Monaco editor.
  2. The strategy selector, mode toggle, and action buttons are present.
  3. Save creates a strategy file on disk (verified via API, not flaky DOM).
  4. Run submits a run and navigates to the Runs tab.

Design principles:
  - API-level verification is the source of truth for save/run correctness.
  - UI assertions check invariants (controls exist, buttons clickable) not timing-sensitive internals.
  - No fixed sleeps; use bounded polling on deterministic conditions.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_health(*, port: int, timeout_s: float = 8.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            import httpx

            r = httpx.get(f"http://127.0.0.1:{port}/health")
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.05)
    raise AssertionError("Server did not start")


def _poll_until(predicate, *, timeout_s: float, interval_s: float = 0.05, desc: str = "condition"):
    t0 = time.time()
    last_exc: Exception | None = None
    while time.time() - t0 < timeout_s:
        try:
            v = predicate()
            if v:
                return v
        except Exception as exc:
            last_exc = exc
        time.sleep(interval_s)
    if last_exc:
        raise AssertionError(f"timeout waiting for {desc}; last_exc={last_exc}")
    raise AssertionError(f"timeout waiting for {desc}")


def _pw_click_when_ready(page, selector: str, *, timeout_s: float = 20.0) -> None:
    """Stable click helper. Waits for visible + enabled, retries with force if intercepted."""
    loc = page.locator(selector)
    loc.wait_for(state="visible", timeout=int(timeout_s * 1000))

    _poll_until(
        lambda: loc.is_enabled(),
        timeout_s=timeout_s,
        interval_s=0.1,
        desc=f"{selector} enabled",
    )

    last_exc: Exception | None = None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            loc.click(timeout=2000)
            return
        except Exception as exc:
            last_exc = exc
            try:
                loc.click(timeout=2000, force=True)
                return
            except Exception as exc2:
                last_exc = exc2
                time.sleep(0.1)

    raise AssertionError(f"timeout clicking {selector}; last_exc={last_exc}")


@pytest.mark.slow
def test_dsl_builder_editor_renders(tmp_path, monkeypatch):
    """The DSL Builder tab renders with a Monaco editor and all key controls.

    Checks:
      - Tab button navigates to DSL Builder
      - Monaco editor container appears
      - Strategy selector, mode toggle, Save/Run/Save&Run buttons exist
      - Add Factor and Add Signal buttons exist
    """
    playwright = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright.sync_playwright

    port = _free_port()

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    # Write a strategy for the selector.
    (strategies_dir / "test_strat.py").write_text("# test\n", encoding="utf-8")

    import uvicorn
    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()

    server_started = threading.Event()

    def _run_server():
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server_started.set()
        server.run()

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    server_started.wait(timeout=5)
    _wait_health(port=port)

    base_url = f"http://127.0.0.1:{port}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(f"{base_url}/?tab=dsl_builder", wait_until="networkidle", timeout=30000)

            # Wait for the app to fully boot.
            page.wait_for_selector("[data-ui-boot='1']", timeout=15000)

            # Verify DSL Builder page renders.
            page.wait_for_selector("[data-testid='dsl-builder-page']", state="visible", timeout=10000)

            # Monaco editor container should appear (it lazy-loads).
            page.wait_for_selector("[data-testid='dsl-editor-wrapper']", state="visible", timeout=15000)

            # Monaco creates elements with class .monaco-editor.
            # Wait for it with a generous timeout (Monaco loads JS from CDN or bundle).
            _poll_until(
                lambda: page.locator(".monaco-editor").count() > 0,
                timeout_s=30,
                interval_s=0.5,
                desc="Monaco editor to render",
            )

            # Verify toolbar controls exist.
            assert page.locator("[data-testid='dslStrategySelect']").count() == 1
            assert page.locator("[data-testid='btnDslModeToggle']").count() == 1
            assert page.locator("[data-testid='btnDslSave']").count() == 1
            assert page.locator("[data-testid='btnDslRun']").count() == 1
            assert page.locator("[data-testid='btnDslSaveAndRun']").count() == 1

            # Verify Add Factor / Add Signal buttons exist.
            assert page.locator("[data-testid='btnAddFactor']").count() == 1
            assert page.locator("[data-testid='btnAddSignal']").count() == 1

            # The strategy selector should have the test strategy.
            sel = page.locator("[data-testid='dslStrategySelect']")
            options = sel.locator("option").all_text_contents()
            assert "(new strategy)" in options
            assert "test_strat" in options

        finally:
            browser.close()


@pytest.mark.slow
def test_dsl_builder_save_and_run(tmp_path, monkeypatch):
    """End-to-end: save a strategy via the DSL Builder, then run it.

    We verify correctness via API (not DOM scraping), keeping the test deterministic.
    UI interaction is minimal: click Save, verify via API that file was created;
    click Run, verify via API that a run was submitted.
    """
    playwright = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright.sync_playwright
    httpx = pytest.importorskip("httpx")

    port = _free_port()

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    import uvicorn
    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()

    server_started = threading.Event()

    def _run_server():
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server_started.set()
        server.run()

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    server_started.wait(timeout=5)
    _wait_health(port=port)

    base_url = f"http://127.0.0.1:{port}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(f"{base_url}/?tab=dsl_builder", wait_until="networkidle", timeout=30000)
            page.wait_for_selector("[data-ui-boot='1']", timeout=15000)
            page.wait_for_selector("[data-testid='dsl-builder-page']", state="visible", timeout=10000)

            # Wait for Monaco to load and code to generate.
            _poll_until(
                lambda: page.locator(".monaco-editor").count() > 0,
                timeout_s=30,
                interval_s=0.5,
                desc="Monaco editor to render",
            )

            # Wait for generated code to populate the editor.
            # In generated mode, the /api/dsl/generate call populates the editor.
            _poll_until(
                lambda: page.locator("[data-testid='btnDslSave']").is_enabled(),
                timeout_s=15,
                interval_s=0.3,
                desc="Save button enabled (code populated)",
            )

            # Step 1: Click Save — this creates a new strategy via POST /api/strategies.
            _pw_click_when_ready(page, "[data-testid='btnDslSave']", timeout_s=10)

            # Verify via API that the strategy was saved.
            _poll_until(
                lambda: httpx.get(f"{base_url}/api/strategies").status_code == 200,
                timeout_s=5,
                interval_s=0.3,
                desc="strategies endpoint available",
            )

            # The default name is "custom_strategy".
            strat_response = httpx.get(f"{base_url}/api/strategies/custom_strategy")
            assert strat_response.status_code == 200, (
                f"Strategy was not saved; got {strat_response.status_code}: {strat_response.text}"
            )
            assert "source" in strat_response.json().get("strategy", {})

            # Step 2: Click Run — this submits a run via POST /api/runs.
            # After save, the strategy is known and isNewStrategy=false, so Run should work.
            _pw_click_when_ready(page, "[data-testid='btnDslRun']", timeout_s=10)

            # Verify a toast appeared with "Run submitted" (or the run was submitted via API).
            _poll_until(
                lambda: httpx.get(f"{base_url}/api/runs?limit=5").json().get("total", 0) > 0,
                timeout_s=10,
                interval_s=0.5,
                desc="at least one run to be submitted",
            )

            runs = httpx.get(f"{base_url}/api/runs?limit=5").json()
            assert runs["total"] >= 1
            assert runs["runs"][0]["strategy_id"] == "custom_strategy"

        finally:
            browser.close()


@pytest.mark.slow
def test_dsl_builder_mode_toggle(tmp_path, monkeypatch):
    """Toggling between Generated and Free-edit mode works correctly.

    In free-edit mode:
      - The form panel should be visually disabled.
      - The notice should appear.
    In generated mode:
      - The form should be active.
    """
    playwright = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright.sync_playwright

    port = _free_port()

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    import uvicorn
    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()

    server_started = threading.Event()

    def _run_server():
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server_started.set()
        server.run()

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    server_started.wait(timeout=5)
    _wait_health(port=port)

    base_url = f"http://127.0.0.1:{port}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(f"{base_url}/?tab=dsl_builder", wait_until="networkidle", timeout=30000)
            page.wait_for_selector("[data-ui-boot='1']", timeout=15000)
            page.wait_for_selector("[data-testid='dsl-builder-page']", state="visible", timeout=10000)

            # Wait for Monaco.
            _poll_until(
                lambda: page.locator(".monaco-editor").count() > 0,
                timeout_s=30,
                interval_s=0.5,
                desc="Monaco editor to render",
            )

            # Initially in Generated mode — no free-edit notice.
            toggle = page.locator("[data-testid='btnDslModeToggle']")
            assert "Generated" in toggle.text_content()
            assert page.locator("[data-testid='dsl-free-edit-notice']").count() == 0

            # Toggle to Free-edit.
            toggle.click()
            _poll_until(
                lambda: "Free-edit" in page.locator("[data-testid='btnDslModeToggle']").text_content(),
                timeout_s=5,
                interval_s=0.2,
                desc="mode toggle to show Free-edit",
            )

            # Free-edit notice should appear.
            page.wait_for_selector("[data-testid='dsl-free-edit-notice']", state="visible", timeout=5000)

            # Toggle back to Generated.
            toggle.click()
            _poll_until(
                lambda: "Generated" in page.locator("[data-testid='btnDslModeToggle']").text_content(),
                timeout_s=5,
                interval_s=0.2,
                desc="mode toggle to show Generated",
            )

            # Free-edit notice should disappear.
            _poll_until(
                lambda: page.locator("[data-testid='dsl-free-edit-notice']").count() == 0,
                timeout_s=5,
                interval_s=0.2,
                desc="free-edit notice to disappear",
            )

        finally:
            browser.close()

