#!/usr/bin/env python3
"""
GLTF Module Logging

Provides logging functionality compatible with Blender's logging system.
Uses Python's standard logging module for consistency.
"""

import logging
import sys
from typing import Optional


# Create the main GLTF logger
logger = logging.getLogger('gltf_module')
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if module is imported multiple times
if not logger.handlers:
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific GLTF module component.

    Args:
        name: Component name (e.g., 'document', 'exporter', 'decoder')

    Returns:
        Logger instance for the component
    """
    return logging.getLogger(f'gltf_module.{name}')


def set_log_level(level: int):
    """
    Set the log level for the GLTF module.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def setup_blender_logging():
    """
    Configure logging to work well with Blender's logging system.
    This ensures compatibility when the module is used within Blender.
    """
    # In Blender, we want to integrate with Blender's logging
    # Blender typically uses its own logging handlers
    try:
        import bpy
        # If we're in Blender, let Blender handle the logging
        # Blender's console will show our log messages
        pass
    except ImportError:
        # Not in Blender, use our default setup
        pass


# Initialize Blender logging compatibility
setup_blender_logging()


# Convenience functions for common logging patterns
def log_info(message: str, component: Optional[str] = None):
    """Log an info message."""
    if component:
        get_logger(component).info(message)
    else:
        logger.info(message)


def log_warning(message: str, component: Optional[str] = None):
    """Log a warning message."""
    if component:
        get_logger(component).warning(message)
    else:
        logger.warning(message)


def log_error(message: str, component: Optional[str] = None):
    """Log an error message."""
    if component:
        get_logger(component).error(message)
    else:
        logger.error(message)


def log_debug(message: str, component: Optional[str] = None):
    """Log a debug message."""
    if component:
        get_logger(component).debug(message)
    else:
        logger.debug(message)
