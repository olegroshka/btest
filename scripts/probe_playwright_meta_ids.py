import socket
import threading
import time

import httpx
import uvicorn
from playwright.sync_api import sync_playwright

from quantdsl_backtest.platform_api.main import app


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def wait_ready(port: int, timeout_s: float = 10.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.05)
    raise RuntimeError("server not ready")


def main() -> None:
    port = get_free_port()

    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning"),
        daemon=True,
    ).start()

    wait_ready(port)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/?tab=meta", wait_until="networkidle")

        ids = page.evaluate(
            """() => Array.from(document.querySelectorAll('#pageMeta [id]')).map(e => e.id)"""
        )
        print("meta ids:", ids)

        try:
            print("mEntity value:", page.eval_on_selector("#mEntity", "el => el.value"))
        except Exception as e:
            print("no mEntity:", repr(e))

        browser.close()


if __name__ == "__main__":
    main()

