"""Logging helpers with structured JSON output and rotation."""

import json
import sys
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - thin wrapper
        """Serialize one log record as a JSON object.

        Args:
            record (logging.LogRecord): Runtime log record to format.

        Returns:
            str: JSON string containing base fields plus optional context.
        """
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if isinstance(getattr(record, "context", None), dict):
            payload.update(record.context)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


class ContextTextFormatter(logging.Formatter):
    """Plain-text formatter that appends structured context when present."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - thin wrapper
        """Render a text log line and append pretty context when available.

        Args:
            record (logging.LogRecord): Runtime log record to format.

        Returns:
            str: Text log line with optional formatted context block.
        """
        base = super().format(record)
        context = getattr(record, "context", None)
        if isinstance(context, dict) and context:
            compact_context = _compact_for_log(context)
            pretty_context = json.dumps(
                compact_context,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                default=str,
            )
            return f"{base}\ncontext:\n{pretty_context}"
        return base


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging with stream output and optional rotating file output."""

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handlers: list[logging.Handler] = [stream_handler]

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            ContextTextFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    _tune_noisy_loggers()


def _ensure_rotating_file_handler(
    logger: logging.Logger,
    log_file: Path,
    max_bytes: int,
    backup_count: int,
) -> None:
    """Attach a rotating file handler when one is not already configured.

    Args:
        logger (logging.Logger): Logger to configure.
        log_file (Path): Target log file path.
        max_bytes (int): Maximum file size before rotation.
        backup_count (int): Number of rotated files to keep.

    Returns:
        None: Mutates logger handlers in place.
    """
    existing = [
        handler
        for handler in logger.handlers
        if isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == log_file.resolve()
    ]
    if existing:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        ContextTextFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


def _purge_expired_logs(log_file: Path, retention_days: Optional[int]) -> None:
    """Delete expired rotated log files according to retention policy.

    Args:
        log_file (Path): Base log file used to discover rotated siblings.
        retention_days (Optional[int]): Number of days to retain log files.

    Returns:
        None: Removes expired files when possible.
    """
    if not retention_days or retention_days <= 0:
        return

    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    for candidate in log_file.parent.glob(f"{log_file.name}*"):
        try:
            if datetime.utcfromtimestamp(candidate.stat().st_mtime) < cutoff:
                candidate.unlink(missing_ok=True)
        except OSError:
            continue


def _ensure_stream_handler(logger: logging.Logger, level: str) -> None:
    """Ensure a stdout stream handler exists with the requested level.

    Args:
        logger (logging.Logger): Logger to configure.
        level (str): Log level name for stream output.

    Returns:
        None: Updates or adds a stream handler.
    """
    existing = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]
    if existing:
        existing[0].setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(stream_handler)


def get_structured_logger(
    name: str,
    log_file: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    retention_days: int = 7,
    level: str = "DEBUG",
    stream_level: str = "INFO",
) -> logging.Logger:
    """Return a logger configured for JSON file output with rotation and stdout output."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    logger.propagate = False

    _ensure_rotating_file_handler(logger, Path(log_file), max_bytes, backup_count)
    _ensure_stream_handler(logger, stream_level)
    _purge_expired_logs(Path(log_file), retention_days)
    _tune_noisy_loggers()

    return logger


@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """Return standard logger without structured handlers."""

    return logging.getLogger(name)


def _compact_for_log(
    value,
    *,
    max_str: int = 220,
    max_items: int = 12,
    max_depth: int = 5,
):
    """Limit deeply nested/large logging payloads to readable sizes."""

    if max_depth <= 0:
        return "<max-depth-reached>"

    if isinstance(value, str):
        if len(value) <= max_str:
            return value
        return f"{value[:max_str]}...(+{len(value) - max_str} chars)"

    if isinstance(value, dict):
        output = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            output[str(key)] = _compact_for_log(
                item,
                max_str=max_str,
                max_items=max_items,
                max_depth=max_depth - 1,
            )
        if len(items) > max_items:
            output["__truncated_keys__"] = len(items) - max_items
        return output

    if isinstance(value, list):
        compact_list = [
            _compact_for_log(
                item,
                max_str=max_str,
                max_items=max_items,
                max_depth=max_depth - 1,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            compact_list.append(f"...(+{len(value) - max_items} items)")
        return compact_list

    return value


def _tune_noisy_loggers() -> None:
    """Reduce noisy library logs so app-level structured logs stay readable."""

    noisy_levels = {
        "httpcore": logging.WARNING,
        "httpx": logging.INFO,
        "openai._base_client": logging.INFO,
        "urllib3.connectionpool": logging.INFO,
    }
    for logger_name, level in noisy_levels.items():
        logging.getLogger(logger_name).setLevel(level)
