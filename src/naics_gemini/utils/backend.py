#!/usr/bin/env python3

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import platform
import sys

import torch

# -------------------------------------------------------------------------------------------------
# Backend GPU availability tests
# -------------------------------------------------------------------------------------------------

def main():

    print(f'Python {sys.version.split()[0]} on {platform.system()} {platform.processor()}')
    print(f'Torch version: {torch.__version__}')

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    if cuda_ok:
        print(f'CUDA available: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA version: {torch.version.cuda}') # type: ignore
    elif mps_ok:
        print('MPS (Apple Silicon Metal) available')

    else:
        print('No GPU backend detected.')
        print('  Torch compiled with:')
        print(f'   CUDA: {torch.version.cuda}')  # type: ignore
        print(f"   MPS backend: {hasattr(torch.backends, 'mps')}")

    x = torch.randn(2, 2)
    device = 'cuda' if cuda_ok else 'mps' if mps_ok else 'cpu'
    x = x.to(device)
    
    print(f'Ran tensor test on: {device.upper()} — success!')

if __name__ == '__main__':
    main()
