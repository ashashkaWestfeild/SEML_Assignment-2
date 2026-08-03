"""Structured (JSON) logging used across every module.

Why JSON: the service is expected to run behind a log-aggregation agent
(ELK / CloudWatch / Splunk). Line-oriented JSON is machine-parseable, so the
same records that help a developer debug locally become the monitoring
substrate in production without a second instrumentation path.

Log-level policy used consistently in this codebase:
    INFO     - normal lifecycle events worth auditing (data loaded, model
               trained, prediction served).
    WARNING  - recoverable anomalies: a fallback was taken, a soft data-quality
               rule was breached, a business rule rejected a request.
    ERROR    - the operation failed and the caller gets an error response.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from types import TracebackType
from typing import Any, Dict, Optional

_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
    "message",
}


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` (plus any ``extra=`` fields) as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a module-scoped logger with exactly one JSON handler attached."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class Timer:
    """Context manager measuring wall-clock duration in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
