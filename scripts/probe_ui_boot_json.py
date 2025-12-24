from __future__ import annotations

import json
import pathlib


def main() -> None:
    p = pathlib.Path(".tmp_probe_ui_boot.json")
    raw = p.read_bytes().replace(b"\x00", b"")

    txt = None
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            t = raw.decode(enc)
            if '"paths"' in t:
                txt = t
                break
        except Exception:
            continue

    if txt is None:
        raise SystemExit("could not decode")

    obj = json.loads(txt)
    bad = {k: v for k, v in obj.get("paths", {}).items() if v.get("status") != 200}
    print(json.dumps({"bad": bad, "base": obj.get("base")}, indent=2))


if __name__ == "__main__":
    main()

