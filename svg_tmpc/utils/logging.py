# svg_tmpc/utils/logging.py
"""Structured logging helpers built on the stdlib ``logging`` module."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str = "INFO", stream=None) -> None:
    """Install a single stream handler at the root logger.

    Idempotent: repeated calls do not stack handlers.
    """
    root = logging.getLogger()
    if getattr(root, "_svg_tmpc_configured", False):
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root._svg_tmpc_configured = True  # type: ignore[attr-defined]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not getattr(logging.getLogger(), "_svg_tmpc_configured", False):
        configure_logging()
    return logging.getLogger(name if name else "svg_tmpc")
