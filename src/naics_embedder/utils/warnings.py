# -------------------------------------------------------------------------------------------------
# Centralized Warning Configuration
# -------------------------------------------------------------------------------------------------
'''
Centralized warning suppression for NAICS Embedder.

This module provides a single location for configuring warning filters across the
entire package. Import and call ``configure_warnings()`` at application startup to
apply consistent suppression rules.

Rationale for Suppressed Warnings:
----------------------------------

1. **Precision not supported by model summary** (PyTorch Lightning)
   - Occurs when using mixed precision; the summary still works correctly
   - Suppressed to reduce noise during training startup

2. **Module in eval mode** (PyTorch Lightning)
   - Warning about frozen modules during training; expected behavior with LoRA
   - Our architecture intentionally keeps some modules frozen

3. **DataLoader has few workers** (PyTorch Lightning)
   - Triggered on systems with limited resources; not actionable in many cases
   - Users can adjust num_workers in config if needed

4. **Checkpoint directory exists and is not empty** (PyTorch Lightning)
   - Expected when resuming training or running multiple experiments
   - Suppressed to keep output clean during normal operation

5. **Trying to infer batch_size** (PyTorch Lightning)
   - Occurs with custom data structures; does not affect functionality
   - Our collate function returns proper batch structure

Usage:
------

.. code-block:: python

    from naics_embedder.utils.warnings import configure_warnings

    # Apply all warning filters at startup
    configure_warnings()
'''

import warnings
from typing import List, Optional, Tuple

# -------------------------------------------------------------------------------------------------
# Warning Definitions
# -------------------------------------------------------------------------------------------------

# Each tuple: (message_pattern, category, module_pattern, rationale)
_WARNING_FILTERS: List[Tuple[str, type, str, str]] = [
    (
        '.*Precision.*is not supported by the model summary.*',
        UserWarning,
        'pytorch_lightning.utilities.model_summary.model_summary',
        'Mixed precision summary works correctly despite this warning',
    ),
    (
        '.*Found .* module.*in eval mode.*',
        UserWarning,
        'pytorch_lightning',
        'Expected behavior with LoRA - some modules are intentionally frozen',
    ),
    (
        '.*does not have many workers.*',
        UserWarning,
        'pytorch_lightning',
        'Resource-dependent; users can adjust num_workers in config',
    ),
    (
        '.*Checkpoint directory.*exists and is not empty.*',
        UserWarning,
        'pytorch_lightning',
        'Expected when resuming training or running experiments',
    ),
    (
        '.*Trying to infer the.*batch_size.*',
        UserWarning,
        'pytorch_lightning',
        'Custom collate returns proper structure; inference not needed',
    ),
]

# -------------------------------------------------------------------------------------------------
# Configuration Function
# -------------------------------------------------------------------------------------------------

def configure_warnings(
    additional_filters: Optional[List[Tuple[str, type, str]]] = None, verbose: bool = False
) -> None:
    '''
    Configure warning filters for the NAICS Embedder application.

    Applies the standard set of warning suppressions to reduce noise during
    training while preserving meaningful warnings from other sources.

    Args:
        additional_filters: Optional list of additional warning filters to apply.
            Each tuple should contain (message_pattern, category, module_pattern).
        verbose: If True, log each warning filter as it is applied.

    Example:
        >>> from naics_embedder.utils.warnings import configure_warnings
        >>> configure_warnings()  # Apply standard filters

        >>> # Add custom filter
        >>> configure_warnings(additional_filters=[('.*my custom warning.*', UserWarning, 'my_module')])
    '''
    import logging

    logger = logging.getLogger(__name__)

    # Apply standard filters
    for message, category, module, rationale in _WARNING_FILTERS:
        warnings.filterwarnings('ignore', message=message, category=category, module=module)
        if verbose:
            logger.debug(f'Suppressed warning: {message[:50]}... ({rationale})')

    # Apply additional filters if provided
    if additional_filters:
        for message, category, module in additional_filters:
            warnings.filterwarnings('ignore', message=message, category=category, module=module)
            if verbose:
                logger.debug(f'Suppressed additional warning: {message[:50]}...')

def get_warning_rationale(pattern: str) -> Optional[str]:
    '''
    Get the rationale for a specific warning suppression.

    Args:
        pattern: The message pattern to look up.

    Returns:
        The rationale string if found, None otherwise.

    Example:
        >>> rationale = get_warning_rationale('.*Precision.*')
        >>> print(rationale)
        'Mixed precision summary works correctly despite this warning'
    '''
    for message, _, _, rationale in _WARNING_FILTERS:
        if pattern in message or message in pattern:
            return rationale
    return None

def list_suppressed_warnings() -> List[Tuple[str, str]]:
    '''
    List all suppressed warnings and their rationales.

    Returns:
        List of (pattern, rationale) tuples for documentation purposes.

    Example:
        >>> for pattern, rationale in list_suppressed_warnings():
        ...     print(f'{pattern}: {rationale}')
    '''
    return [(msg, rationale) for msg, _, _, rationale in _WARNING_FILTERS]
