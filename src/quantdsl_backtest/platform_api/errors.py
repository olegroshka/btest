from __future__ import annotations

from typing import Any


def to_api_error(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    status: int | None = None,
    request_id: str | None = None,
) -> dict:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "details": details,
    }
    if status is not None:
        error["status"] = int(status)
    if request_id is not None:
        error["request_id"] = str(request_id)
    return {"error": error}
