# src/quantdsl_backtest/utils/logging.py

from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(name)s] "
    "%(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a simple stdout handler if it
    hasn't been configured yet.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured by the application / tests
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger for the given name, ensuring the root logger is configured.

    Usage:
        from quantdsl_backtest.utils.logging import get_logger

        log = get_logger(__name__)
        log.info("Hello")
    """
    _configure_root_logger(level=level)
    return logging.getLogger(name)
