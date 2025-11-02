# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------
 
import logging

from rich.logging import RichHandler

# -------------------------------------------------------------------------------------------------
# Configure logging
# -------------------------------------------------------------------------------------------------
 
def configure_logging(level='INFO'):

    '''
    Configures logging to use RichHandler for beautiful output.
    '''
    
    from rich.console import Console
    console = Console(markup=False)
    
    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_suppress=[
                    'typer', 
                    'click', 
                    'hydra',
                    'pytorch_lightning',
                    'torch'
                ],
                show_path=False,
                console=console,
                show_time=True,
                show_level=False,
                markup=True,
                log_time_format='[%X]'
            )
        ],
    )
    
    # Set lower levels for noisy libraries
    logging.getLogger('hydra').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    # Re-enable our own logger
    logging.getLogger('naics_gemini').setLevel(level)