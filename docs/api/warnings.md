# Warnings Configuration

Centralized warning suppression for the NAICS Embedder package.

## Overview

The `warnings` module provides a single location for configuring warning filters
across the entire package. This centralizes PyTorch Lightning and other library
warnings that would otherwise be scattered across multiple modules.

## Usage

Import and call `configure_warnings()` at application startup:

```python
from naics_embedder.utils.warnings import configure_warnings

# Apply standard warning filters
configure_warnings()
```

## API Reference

::: naics_embedder.utils.warnings
    options:
      show_source: false
      members:
        - configure_warnings
        - get_warning_rationale
        - list_suppressed_warnings

