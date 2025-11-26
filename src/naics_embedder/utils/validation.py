# -------------------------------------------------------------------------------------------------
# Data and Configuration Validation
# -------------------------------------------------------------------------------------------------
'''
Pre-flight validation utilities for NAICS Embedder.

This module provides validation functions that check data files, tokenization
caches, and configuration consistency before training or embedding generation
begins. Early validation prevents runtime surprises from missing files or
incompatible data.

Functions:
    validate_data_paths: Verify required data files exist and are accessible.
    validate_parquet_schema: Check parquet file has expected columns.
    validate_tokenization_cache: Verify tokenization cache compatibility.
    validate_training_config: Comprehensive pre-flight validation for training.
    ValidationError: Exception for validation failures with remediation steps.
'''

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import polars as pl

from naics_embedder.utils.config import Config, TokenizationConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Custom Exceptions
# -------------------------------------------------------------------------------------------------

class ValidationError(Exception):
    '''
    Exception raised when validation fails.

    Includes actionable remediation steps to help users fix the issue.

    Attributes:
        message: Description of the validation failure.
        remediation: List of suggested steps to fix the issue.
        details: Optional additional details about the failure.
    '''

    def __init__(
        self, message: str, remediation: Optional[List[str]] = None, details: Optional[str] = None
    ):
        self.message = message
        self.remediation = remediation or []
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.details:
            parts.append(f'\nDetails: {self.details}')
        if self.remediation:
            parts.append('\nRemediation steps:')
            for i, step in enumerate(self.remediation, 1):
                parts.append(f'  {i}. {step}')
        return '\n'.join(parts)

# -------------------------------------------------------------------------------------------------
# Validation Result
# -------------------------------------------------------------------------------------------------

@dataclass
class ValidationResult:
    '''
    Result of a validation check.

    Attributes:
        valid: Whether validation passed.
        errors: List of error messages.
        warnings: List of warning messages.
    '''

    valid: bool
    errors: List[str]
    warnings: List[str]

    @classmethod
    def success(cls) -> 'ValidationResult':
        '''Create a successful validation result.'''
        return cls(valid=True, errors=[], warnings=[])

    @classmethod
    def failure(cls, error: str) -> 'ValidationResult':
        '''Create a failed validation result with an error message.'''
        return cls(valid=False, errors=[error], warnings=[])

    def add_error(self, error: str) -> None:
        '''Add an error and mark as invalid.'''
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        '''Add a warning (does not affect validity).'''
        self.warnings.append(warning)

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        '''Merge another result into this one.'''
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False
        return self

# -------------------------------------------------------------------------------------------------
# Data Path Validation
# -------------------------------------------------------------------------------------------------

def validate_data_paths(cfg: Config) -> ValidationResult:
    '''
    Verify that required data files exist and are accessible.

    Checks for the existence of description, distance, relation, and triplet
    parquet files required for training.

    Args:
        cfg: Configuration containing data paths.

    Returns:
        ValidationResult indicating success or listing missing files.

    Example:
        >>> result = validate_data_paths(cfg)
        >>> if not result.valid:
        ...     for error in result.errors:
        ...         print(f'Missing: {error}')
    '''
    result = ValidationResult.success()

    required_files = {
        'descriptions': cfg.data_loader.streaming.descriptions_parquet,
        'distances': cfg.data_loader.streaming.distances_parquet,
        'distance_matrix': cfg.data_loader.streaming.distance_matrix_parquet,
        'relations': cfg.data_loader.streaming.relations_parquet,
    }

    for name, path_str in required_files.items():
        path = Path(path_str)
        if not path.exists():
            result.add_error(
                f'{name.capitalize()} file not found: {path_str}\n'
                f'  Run: uv run naics-embedder data all'
            )

    # Check triplets directory
    triplets_path = Path(cfg.data_loader.streaming.triplets_parquet)
    if triplets_path.is_dir():
        parquet_files = list(triplets_path.glob('*.parquet'))
        if not parquet_files:
            result.add_error(
                f'Triplets directory is empty: {triplets_path}\n'
                f'  Run: uv run naics-embedder data triplets'
            )
    elif not triplets_path.exists():
        result.add_error(
            f'Triplets path not found: {triplets_path}\n  Run: uv run naics-embedder data triplets'
        )

    return result

# -------------------------------------------------------------------------------------------------
# Parquet Schema Validation
# -------------------------------------------------------------------------------------------------

def validate_parquet_schema(
    path: str, required_columns: Set[str], file_description: str = 'parquet file'
) -> ValidationResult:
    '''
    Check that a parquet file has the expected columns.

    Args:
        path: Path to the parquet file.
        required_columns: Set of column names that must be present.
        file_description: Human-readable description for error messages.

    Returns:
        ValidationResult with schema validation status.

    Example:
        >>> result = validate_parquet_schema(
        ...     'data/naics_descriptions.parquet',
        ...     {'index', 'code', 'title', 'description'},
        ...     'descriptions',
        ... )
    '''
    result = ValidationResult.success()

    file_path = Path(path)
    if not file_path.exists():
        result.add_error(f'{file_description} not found: {path}')
        return result

    try:
        # Read just the schema without loading data
        schema = pl.read_parquet_schema(path)
        actual_columns = set(schema.keys())

        missing_columns = required_columns - actual_columns
        if missing_columns:
            result.add_error(
                f'{file_description} missing columns: {sorted(missing_columns)}\n'
                f'  Expected: {sorted(required_columns)}\n'
                f'  Found: {sorted(actual_columns)}\n'
                f'  Regenerate with: uv run naics-embedder data all'
            )

    except Exception as e:
        result.add_error(f'Failed to read {file_description} schema: {e}')

    return result

def validate_descriptions_schema(cfg: Config) -> ValidationResult:
    '''
    Validate the descriptions parquet has required columns.

    Args:
        cfg: Configuration with data paths.

    Returns:
        ValidationResult for descriptions schema.
    '''
    return validate_parquet_schema(
        cfg.data_loader.streaming.descriptions_parquet,
        {'index', 'code', 'level', 'title', 'description'},
        'descriptions parquet',
    )

def validate_distances_schema(cfg: Config) -> ValidationResult:
    '''
    Validate the distances parquet has required columns.

    Args:
        cfg: Configuration with data paths.

    Returns:
        ValidationResult for distances schema.
    '''
    return validate_parquet_schema(
        cfg.data_loader.streaming.distances_parquet,
        {'idx_i', 'idx_j', 'distance'},
        'distances parquet',
    )

# -------------------------------------------------------------------------------------------------
# Tokenization Cache Validation
# -------------------------------------------------------------------------------------------------

def validate_tokenization_cache(
    cfg: Config, tokenization_cfg: Optional[TokenizationConfig] = None
) -> ValidationResult:
    '''
    Verify that the tokenization cache exists and is compatible.

    Checks that the cache file exists and was generated with compatible
    settings (tokenizer name, max length).

    Args:
        cfg: Main configuration.
        tokenization_cfg: Optional tokenization-specific config.

    Returns:
        ValidationResult with cache validation status and regeneration
        instructions if needed.

    Example:
        >>> result = validate_tokenization_cache(cfg)
        >>> if not result.valid:
        ...     print('Cache needs regeneration')
    '''
    result = ValidationResult.success()

    if tokenization_cfg is None:
        tokenization_cfg = TokenizationConfig(
            descriptions_parquet=cfg.data_loader.streaming.descriptions_parquet,
            tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
            max_length=cfg.data_loader.tokenization.max_length,
        )

    cache_path = Path(tokenization_cfg.output_path)

    if not cache_path.exists():
        result.add_warning(
            f'Tokenization cache not found: {cache_path}\n'
            f'  Cache will be built on first training run (may take a few minutes)'
        )
        return result

    # Check cache is not empty/corrupted
    try:
        import torch

        cache = torch.load(cache_path, weights_only=True, map_location='cpu')

        if not cache or len(cache) == 0:
            result.add_error(
                f'Tokenization cache is empty: {cache_path}\n'
                f'  Delete and regenerate: rm {cache_path}'
            )
            return result

        # Check cache has expected structure
        sample_key = next(iter(cache.keys()))
        sample_value = cache[sample_key]

        expected_keys = {'code', 'title', 'description', 'excluded', 'examples'}
        actual_keys = set(sample_value.keys())

        if not expected_keys.issubset(actual_keys):
            missing = expected_keys - actual_keys
            result.add_error(
                f'Tokenization cache has wrong structure. Missing keys: {missing}\n'
                f'  Delete and regenerate: rm {cache_path}'
            )

    except Exception as e:
        result.add_error(
            f'Failed to load tokenization cache: {e}\n  Delete and regenerate: rm {cache_path}'
        )

    return result

# -------------------------------------------------------------------------------------------------
# Comprehensive Training Validation
# -------------------------------------------------------------------------------------------------

def validate_training_config(cfg: Config) -> ValidationResult:
    '''
    Run comprehensive pre-flight validation for training.

    Checks all data files, schemas, and configuration settings required
    for a successful training run.

    Args:
        cfg: Training configuration to validate.

    Returns:
        ValidationResult with all validation checks combined.

    Raises:
        ValidationError: If critical validation errors are found and
            ``raise_on_error=True``.

    Example:
        >>> result = validate_training_config(cfg)
        >>> if not result.valid:
        ...     for error in result.errors:
        ...         logger.error(error)
        ...     raise SystemExit(1)
    '''
    result = ValidationResult.success()

    logger.info('Running pre-flight validation...')

    # Validate data paths exist
    result.merge(validate_data_paths(cfg))

    # Validate parquet schemas (only if files exist)
    desc_path = Path(cfg.data_loader.streaming.descriptions_parquet)
    if desc_path.exists():
        result.merge(validate_descriptions_schema(cfg))

    dist_path = Path(cfg.data_loader.streaming.distances_parquet)
    if dist_path.exists():
        result.merge(validate_distances_schema(cfg))

    # Validate tokenization cache
    result.merge(validate_tokenization_cache(cfg))

    # Log results
    if result.valid:
        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)
        logger.info('Pre-flight validation passed')
    else:
        for error in result.errors:
            logger.error(error)
        for warning in result.warnings:
            logger.warning(warning)

    return result

def require_valid_config(cfg: Config) -> None:
    '''
    Validate config and raise if validation fails.

    Convenience function that validates the configuration and raises
    a ValidationError with remediation steps if validation fails.

    Args:
        cfg: Configuration to validate.

    Raises:
        ValidationError: If any validation checks fail.

    Example:
        >>> require_valid_config(cfg)  # Raises on failure
    '''
    result = validate_training_config(cfg)

    if not result.valid:
        raise ValidationError(
            'Training configuration validation failed',
            remediation=[
                'Run: uv run naics-embedder data all',
                'Check that all required data files exist',
                'Verify configuration paths are correct',
            ],
            details='\n'.join(result.errors),
        )
