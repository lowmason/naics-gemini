'''
Utility modules for NAICS Embedder.

This package provides shared utilities for configuration, logging, device
management, and warning suppression.

Modules:
    backend: Device detection and GPU memory utilities.
    config: Pydantic configuration models for training and data.
    console: Rich console logging and table formatting.
    utilities: General helper functions for data and file operations.
    warnings: Centralized warning suppression configuration.
'''

from naics_embedder.utils.backend import get_device
from naics_embedder.utils.config import Config, load_config
from naics_embedder.utils.console import configure_logging
from naics_embedder.utils.warnings import configure_warnings

__all__ = [
    'get_device',
    'Config',
    'load_config',
    'configure_logging',
    'configure_warnings',
]
