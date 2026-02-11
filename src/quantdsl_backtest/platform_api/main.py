from __future__ import annotations


def create_app():
    try:
        from fastapi import FastAPI
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    from .routes import health, providers
    from .routes import catalog
    from .routes import catalog_meta
    from .routes import catalog_symbol
    from .routes import catalog_download
    from .routes import catalog_plan
    from .routes import catalog_sources
    from .routes import ui
    from .routes import quality
    from .routes import dsl_builder

    app = FastAPI(
        title="quantdsl-backtest platform API",
        version="0.1.0",
        description="Local-first API for data catalog and strategy-centric backtesting platform.",
    )

    # --- error handling -------------------------------------------------
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import HTTPException
    from fastapi.exceptions import RequestValidationError

    from .errors import to_api_error
    from .middleware import generate_request_id, sanitize_request_id

    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):
        import time

        from .logging_utils import RequestLogEvent, log_request

        incoming = request.headers.get("x-request-id")
        rid = sanitize_request_id(incoming) or generate_request_id()
        request.state.request_id = rid

        t0 = time.perf_counter()
        response = await call_next(request)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Always echo request id for easier debugging
        response.headers["X-Request-Id"] = rid

        try:
            log_request(
                RequestLogEvent(
                    request_id=rid,
                    method=str(getattr(request, "method", "")),
                    path=str(getattr(request, "url", request).path if hasattr(getattr(request, "url", None), "path") else getattr(request, "url", "")),
                    status_code=int(getattr(response, "status_code", 0) or 0),
                    duration_ms=float(dt_ms),
                )
            )
        except Exception:
            # never break the API due to logging
            pass

        return response

    def _rid(request: Request) -> str | None:
        return getattr(getattr(request, "state", None), "request_id", None)

    def _method(request: Request) -> str:
        return str(getattr(request, "method", ""))

    def _path(request: Request) -> str:
        try:
            return str(request.url.path)
        except Exception:
            return ""

    def _log_err(request: Request, *, status: int, error_code: str, message: str) -> None:
        try:
            from .logging_utils import ErrorLogEvent, log_error

            log_error(
                ErrorLogEvent(
                    request_id=_rid(request) or "",
                    method=_method(request),
                    path=_path(request),
                    status_code=int(status),
                    error_code=str(error_code),
                    message=str(message),
                )
            )
        except Exception:
            pass

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        from fastapi.encoders import jsonable_encoder

        errs = exc.errors()

        def _loc_has(field: str) -> bool:
            for e in errs:
                loc = e.get("loc")
                if not isinstance(loc, (list, tuple)):
                    continue
                # loc example: ("body", "frequency")
                if any(str(x) == field for x in loc):
                    return True
            return False

        if _loc_has("frequency"):
            code = "INVALID_FREQUENCY"
            message = "Invalid frequency"
        elif _loc_has("kind"):
            code = "INVALID_KIND"
            message = "Invalid kind"
        else:
            code = "VALIDATION_ERROR"
            message = "Validation failed"

        _log_err(request, status=422, error_code=code, message=message)

        return JSONResponse(
            status_code=422,
            content=to_api_error(
                code=code,
                message=message,
                details={"errors": jsonable_encoder(errs)},
                status=422,
                request_id=_rid(request),
            ),
            headers={"X-Request-Id": _rid(request) or ""},
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        status = int(getattr(exc, "status_code", 500) or 500)
        detail = getattr(exc, "detail", None)
        if isinstance(detail, str):
            message = detail
            details = None
        else:
            message = "Request failed"
            details = {"detail": detail}

        _log_err(request, status=status, error_code=f"HTTP_{status}", message=message)

        return JSONResponse(
            status_code=status,
            content=to_api_error(
                code=f"HTTP_{status}",
                message=message,
                details=details,
                status=status,
                request_id=_rid(request),
            ),
            headers={"X-Request-Id": _rid(request) or ""},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        _log_err(request, status=500, error_code="UNHANDLED", message=str(exc))
        return JSONResponse(
            status_code=500,
            content=to_api_error(code="UNHANDLED", message=str(exc), status=500, request_id=_rid(request)),
            headers={"X-Request-Id": _rid(request) or ""},
        )

    # --- routes --------------------------------------------------------
    app.include_router(ui.router)
    app.include_router(health.router)
    app.include_router(providers.router, prefix="/api")
    app.include_router(catalog.router, prefix="/api")
    app.include_router(catalog_meta.router, prefix="/api")
    app.include_router(catalog_symbol.router, prefix="/api")
    app.include_router(catalog_download.router, prefix="/api")
    app.include_router(catalog_plan.router, prefix="/api")
    app.include_router(catalog_sources.router, prefix="/api")
    app.include_router(quality.router, prefix="/api")
    app.include_router(dsl_builder.router, prefix="/api")

    # --- catch-all for browser asset probes (non-API only) -------------
    # Browsers/extensions sometimes probe for optional assets (icons, manifests) and log 404s noisily.
    # We keep API semantics untouched: /api/* routes still return proper 404/422/etc.
    from fastapi.responses import Response
    from fastapi import Request

    @app.middleware("http")
    async def _swallow_non_api_404s(request: Request, call_next):
        resp = await call_next(request)
        try:
            path = str(request.url.path)
        except Exception:
            path = ""
        if resp.status_code == 404 and path and not path.startswith("/api"):
            return Response(status_code=204)
        return resp

    return app


app = create_app()
