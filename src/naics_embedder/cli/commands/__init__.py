# -------------------------------------------------------------------------------------------------
# CLI Commands Package
# -------------------------------------------------------------------------------------------------
'''
CLI command modules for NAICS Embedder.

This package contains command modules organized by domain:
- data: Data generation and preprocessing commands
- tools: Utility tools for configuration, GPU optimization, and metrics
- training: Model training commands
'''

from . import data, tools, training

__all__ = ['data', 'tools', 'training']
