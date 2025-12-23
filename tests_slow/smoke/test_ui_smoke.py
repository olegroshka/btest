from __future__ import annotations

import json
import time
import pytest
from playwright.sync_api import ConsoleMessage, sync_playwright

@pytest.mark.slow
@pytest.mark.smoke
@pytest.mark.manual
def test_ui_clickthrough_smoke() -> None:
    """Manual smoke test: click through the UI.
    
    This test is intended to be run against a live server (default http://127.0.0.1:8000/).
    It is marked as 'manual' and skipped by default in automated runs.
    """
    return main()

def _snap_text(page, selector: str) -> str | None:
    try:
        return page.inner_text(selector)
    except Exception:
        return None


def _snap_value(page, selector: str) -> str | None:
    try:
        return page.eval_on_selector(selector, "e => e.value")
    except Exception:
        return None


def main() -> None:
    out: dict[str, object] = {"steps": [], "errors": []}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            console: list[dict[str, str]] = []

            def _on_console(msg: ConsoleMessage) -> None:
                try:
                    console.append({"type": str(msg.type), "text": str(msg.text)})
                except Exception:
                    pass

            page.on("console", _on_console)

            page.goto("http://127.0.0.1:8000/", wait_until="domcontentloaded")
            page.wait_for_selector("#btnCatalog", timeout=15000)
            time.sleep(0.25)

            steps: list[dict[str, object]] = out["steps"]  # type: ignore[assignment]

            steps.append(
                {
                    "name": "loaded",
                    "title": page.title(),
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            page.click("#btnCatalog")
            page.wait_for_timeout(1500)
            steps.append(
                {
                    "name": "after_catalog",
                    "catalog_text": (_snap_text(page, "#catalog") or "")[:400],
                    "catalog_preview_links": len(page.query_selector_all("a[data-act='preview']")),
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            links = page.query_selector_all("a[data-act='preview']")
            if links:
                links[0].click()
                page.wait_for_timeout(600)

            steps.append(
                {
                    "name": "after_select",
                    "pLib": _snap_value(page, "#pLib"),
                    "pSym": _snap_value(page, "#pSym"),
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            # Switch to Inspector using the visible tab control.
            page.click("#mainTabs [data-tab='inspector']")
            page.wait_for_timeout(600)

            # Ensure the Inspector pane is visible.
            page.wait_for_selector("#pageInspector", state="visible", timeout=10000)

            # Preview button must be visible and enabled.
            page.wait_for_selector("#btnPreview", state="visible", timeout=10000)
            preview_disabled = page.is_disabled("#btnPreview")
            steps.append({"name": "inspector_open", "preview_disabled": preview_disabled})

            if not preview_disabled:
                page.locator("#btnPreview").scroll_into_view_if_needed()
                page.click("#btnPreview")
                page.wait_for_timeout(2500)

            steps.append(
                {
                    "name": "after_preview",
                    "summary": (_snap_text(page, "#previewSummary") or "")[:250],
                    "preview_error": (_snap_text(page, "#previewError") or "")[:250],
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            # Switch back to Catalog and query meta
            page.click("#mainTabs [data-tab='catalog']")
            page.wait_for_timeout(400)
            page.click("#btnMeta")
            page.wait_for_timeout(1200)

            steps.append(
                {
                    "name": "after_meta",
                    "meta_text": (_snap_text(page, "#meta") or "")[:400],
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            out["console_tail"] = console[-50:]

            page.wait_for_timeout(600)
            browser.close()
    except Exception as e:
        errs: list[str] = out["errors"]  # type: ignore[assignment]
        errs.append(repr(e))

    with open("ui_clickthrough_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

