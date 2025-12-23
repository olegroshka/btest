from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger("quantdsl_backtest.platform_api")


@dataclass(frozen=True, slots=True)
class RequestLogEvent:
    request_id: str
    method: str
    path: str
    status_code: int
    duration_ms: float

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class ErrorLogEvent:
    request_id: str
    method: str
    path: str
    status_code: int
    error_code: str
    message: str

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "message": self.message,
        }


def log_request(event: RequestLogEvent) -> None:
    # Keep it simple: structured-ish dict in message. Can be swapped for json logger later.
    logger.info("request", extra={"event": event.as_dict()})


def log_error(event: ErrorLogEvent) -> None:
    # Error logs are important for UI debugging; include request_id and stable error_code.
    logger.warning("error", extra={"event": event.as_dict()})
