from __future__ import annotations

import re
import uuid


_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


def sanitize_request_id(value: str | None) -> str | None:
    """Return a safe request_id or None.

    We restrict length + characters so we can safely echo it in response headers and logs.
    """

    if not value:
        return None
    v = value.strip()
    if not _REQUEST_ID_RE.fullmatch(v):
        return None
    return v


def generate_request_id() -> str:
    """Generate a request id.

    Prefer time-sortable UUIDv7 when available (Python 3.11+ may expose uuid.uuid7),
    otherwise fall back to uuid4.
    """

    uuid7 = getattr(uuid, "uuid7", None)
    if callable(uuid7):
        v = uuid7()
        if isinstance(v, uuid.UUID):
            return v.hex
        return str(v)
    return uuid.uuid4().hex
