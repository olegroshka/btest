import socket
import threading
import time

import httpx
import uvicorn
from playwright.sync_api import sync_playwright

from quantdsl_backtest.platform_api.main import app


def get_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def wait_for_health(port: int) -> None:
    t0 = time.time()
    while time.time() - t0 < 15:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("server did not start")


def main() -> None:
    port = get_free_port()
    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning"),
        daemon=True,
    ).start()
    wait_for_health(port)

    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_page()
        page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")

        # We start on Catalog; #tabCatalog is disabled by design.
        page.wait_for_selector("#catalogSearch", state="visible", timeout=10000)
        page.fill("#catalogSearch", "SPX")
        page.eval_on_selector("#catalogSearch", "el => el.dispatchEvent(new Event('input'))")
        page.click("#btnCatalog")
        page.wait_for_selector("#catalog a[data-act='preview']", timeout=10000)

        spx = page.locator("#catalog a[data-act='preview']", has_text="SPX")
        spx.first.wait_for(state="visible", timeout=10000)

        ps_before = page.eval_on_selector("#pSym", "el=>el.value")

        # Switch to Meta (enabled) to read current mSymbol
        page.click("#tabMeta")
        page.wait_for_selector("#pageMeta", state="visible", timeout=10000)
        ms_before = page.eval_on_selector("#mSymbol", "el=>el.value")

        # Back to Catalog: click Inspector then Catalog? easiest is just set location query.
        page.goto(f"http://127.0.0.1:{port}/?tab=catalog", wait_until="networkidle")
        page.wait_for_selector("#catalog a[data-act='preview']", timeout=10000)
        spx = page.locator("#catalog a[data-act='preview']", has_text="SPX")
        spx.first.wait_for(state="visible", timeout=10000)
        spx.first.click()

        ps_after = page.eval_on_selector("#pSym", "el=>el.value")

        page.goto(f"http://127.0.0.1:{port}/?tab=meta", wait_until="networkidle")
        page.wait_for_selector("#pageMeta", state="visible", timeout=10000)

        page.wait_for_timeout(200)
        ml = page.eval_on_selector("#mLibrary", "el=>el.value")
        ms_after = page.eval_on_selector("#mSymbol", "el=>el.value")

        print("pSym before:", ps_before)
        print("mSymbol before:", ms_before)
        print("pSym after:", ps_after)
        print("mLibrary after:", ml)
        print("mSymbol after:", ms_after)

        page.context.close()


if __name__ == "__main__":
    main()

