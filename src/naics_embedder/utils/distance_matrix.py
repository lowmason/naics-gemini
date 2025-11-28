import logging
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)

def load_distance_submatrix(
    distance_matrix_path: Union[str, Path],
    node_codes: Sequence[str],
) -> torch.Tensor:
    '''
    Load the NAICS ground-truth distance matrix and return the submatrix aligned to node_codes.

    Args:
        distance_matrix_path: Path to ``naics_distance_matrix.parquet``.
        node_codes: Sequence of NAICS codes whose pairwise distances are required.

    Returns:
        Torch tensor of shape ``(len(node_codes), len(node_codes))`` containing the
        tree distances ordered according to ``node_codes``.
    '''

    path = Path(distance_matrix_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'Distance matrix not found: {path}')

    if not node_codes:
        raise ValueError('node_codes must contain at least one NAICS code')

    df = pl.read_parquet(path)

    matrix_codes: List[str] = []
    for col in df.columns:
        if '-code_' not in col:
            continue
        matrix_codes.append(col.split('-code_', 1)[1])

    if not matrix_codes:
        raise ValueError(f'Distance matrix {path} missing code metadata in column names')

    code_to_idx = {code: idx for idx, code in enumerate(matrix_codes)}
    missing = [code for code in node_codes if code not in code_to_idx]
    if missing:
        preview = ', '.join(missing[:5])
        raise ValueError(
            f'{len(missing)} codes missing from distance matrix {path} (e.g., {preview})'
        )

    index_array = np.array([code_to_idx[code] for code in node_codes], dtype=np.int64)
    matrix_np = df.to_numpy()
    subset = matrix_np[np.ix_(index_array, index_array)]
    tensor = torch.from_numpy(subset).float()
    if torch.isnan(tensor).any():
        logger.warning('Distance submatrix contains NaNs; replacing with zeros')
        tensor = torch.nan_to_num(tensor, nan=0.0)
    return tensor
