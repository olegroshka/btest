from __future__ import annotations

import json
import time

import httpx


def main() -> None:
    base = "http://127.0.0.1:8001"

    paths = [
        "/",
        "/static/assets/main.mjs",
        "/static/assets/layout.js",
        "/static/assets/catalog.js",
        "/static/assets/inspector.js",
        "/static/assets/api.js",
        "/static/assets/state.js",
        "/static/assets/download.js",
        "/static/assets/quality.js",
        "/static/assets/workspace.js",
        "/static/plotly.min.js",
    ]

    out = {"base": base, "paths": {}}

    with httpx.Client(timeout=5.0) as client:
        # wait for health
        for _ in range(50):
            try:
                r = client.get(base + "/health")
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("server not healthy")

        for p in paths:
            try:
                r = client.get(base + p)
                out["paths"][p] = {
                    "status": r.status_code,
                    "content_type": r.headers.get("content-type"),
                    "head": (r.text or "")[:200],
                }
            except Exception as e:
                out["paths"][p] = {"error": repr(e)}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

