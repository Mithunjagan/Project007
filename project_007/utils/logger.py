"""
PROJECT 007 — Logging Utility
Configured Python logging with color-coded console output.
"""

import logging
import sys

# ANSI color codes for console output
_COLORS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[41m",    # Red background
    "RESET":    "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """Custom formatter that injects ANSI color codes around the log level."""

    def format(self, record):
        color = _COLORS.get(record.levelname, _COLORS["RESET"])
        reset = _COLORS["RESET"]
        # Color only the level name; leave the rest untouched
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Loggers are created once per name; subsequent calls return the
    same instance (standard logging behaviour).
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = _ColorFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
