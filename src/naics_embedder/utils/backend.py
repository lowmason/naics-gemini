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

def get_device(log_info: bool = False) -> Tuple[str, str, int]:
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    if cuda_ok:
        num_gpus = torch.cuda.device_count()
        gpu = f'  • GPU:\n    - CUDA ({torch.version.cuda})'  # type: ignore
    elif mps_ok:
        num_gpus = 1
        gpu = '  • GPU:\n    - MPS (Apple Silicon Metal)'
    else:
        num_gpus = 0
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

    return device, precision, num_gpus

if __name__ == '__main__':
    device, precision, num_gpus = get_device()
