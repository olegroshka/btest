from __future__ import annotations


def test_platform_api_create_app_importable_without_fastapi_installed():
    """Unit tests should not require platform extras.

    The platform_api package is optional and must not break core imports.
    """
    from quantdsl_backtest.platform_api import main

    # create_app is defined; calling it without extra may raise.
    assert hasattr(main, "create_app")

