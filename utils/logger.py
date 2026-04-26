"""Logging utility for training and evaluation."""

import logging
import os
import sys
from datetime import datetime


def setup_logger(
    name: str = "MSCA-FasterNet",
    log_dir: str = "logs",
    log_file: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logger with console and file output.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        log_file: Log file name. Auto-generated if None.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Format
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
