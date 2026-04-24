"""
DELTA - data/logger.py
Configurazione centralizzata del sistema di logging.
Supporta output su file rotante e console simultaneamente.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from core.config import LOGGING_CONFIG


def setup_logger(name: str = "delta") -> logging.Logger:
    """
    Configura e restituisce il logger root per il sistema DELTA.
    Aggiunge handler console e file rotante se non già presenti.

    Args:
        name: nome del logger (default: 'delta')

    Returns:
        Logger configurato
    """
    logger = logging.getLogger(name)

    # Evita duplicazione handler se già configurato
    if logger.handlers:
        return logger

    level = getattr(logging, LOGGING_CONFIG.get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG["format"],
        datefmt=LOGGING_CONFIG["date_format"],
    )

    # ── Handler console (stdout) ─────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Handler file rotante ─────────────────────────────────
    log_file = Path(LOGGING_CONFIG["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=LOGGING_CONFIG["max_bytes"],
        backupCount=LOGGING_CONFIG["backup_count"],
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
