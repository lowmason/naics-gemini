# Data Validation

Pre-flight validation utilities for data files and configuration.

## Overview

The `validation` module provides functions to check data files, tokenization
caches, and configuration consistency before training begins. Early validation
prevents runtime surprises from missing files or incompatible data.

## Usage

```python
from naics_embedder.utils.validation import (
    validate_training_config,
    require_valid_config,
    ValidationError,
)

# Validate and get result
result = validate_training_config(cfg)
if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")

# Or raise on failure
try:
    require_valid_config(cfg)
except ValidationError as e:
    print(e.message)
    for step in e.remediation:
        print(f"  - {step}")
```

## Validation Checks

The validation system checks:

1. **Data Paths** - Required parquet files exist
2. **Schema Validation** - Parquet files have expected columns
3. **Tokenization Cache** - Cache exists and has correct structure
4. **Configuration Consistency** - Settings are compatible

## API Reference

::: naics_embedder.utils.validation
    options:
      show_source: false
      members:
        - ValidationError
        - ValidationResult
        - validate_data_paths
        - validate_parquet_schema
        - validate_tokenization_cache
        - validate_training_config
        - require_valid_config

