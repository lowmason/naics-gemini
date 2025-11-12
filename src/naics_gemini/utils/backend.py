# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import platform
import sys
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Backend GPU availability tests
# -------------------------------------------------------------------------------------------------

def get_device(log_info: bool = False) -> Tuple[str, str]:

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    if cuda_ok:
        gpu = f'  • GPU:\n    - CUDA ({torch.version.cuda})' #type: ignore
    elif mps_ok:
        gpu = '  • GPU:\n    - MPS (Apple Silicon Metal)'

    else:
        gpu = '  • GPU:\n    - No GPU backend detected, using CPU'
        
    device = 'cuda' if cuda_ok else 'mps' if mps_ok else 'cpu'
    precision = '16-mixed' if cuda_ok else '32-true'

    if log_info:
        logger.info(
            '  • Python:\n'
            f'    - version: {sys.version.split()[0]}\n'
            f'    - platform: {platform.system()} {platform.processor()}'
        )
        logger.info(f'  • Torch:\n    - version: {torch.__version__}')
        logger.info(f'{gpu}\n')

    return device, precision

if __name__ == '__main__':
    device, precision = get_device()
