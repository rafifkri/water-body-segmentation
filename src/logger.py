"""
Logging configuration module
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Path,
    log_name: str = 'training',
    log_level: str = 'INFO',
    log_to_file: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_name: Name of the logger
        log_level: Logging level
        log_to_file: Whether to log to file
    
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_file = log_dir / f'{log_name}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10 MB
            backupCount=10
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
