# -------------------------------------------------------------------------------------------------
# Commands Package
# -------------------------------------------------------------------------------------------------

"""
CLI command modules for NAICS Embedder.

This package organizes CLI commands into logical groups:
- data: Data generation and preprocessing commands
- tools: Utility tools for configuration, GPU optimization, and metrics
- training: Model training commands
"""

from .commands import data, tools, training

__all__ = ['data', 'tools', 'training']
