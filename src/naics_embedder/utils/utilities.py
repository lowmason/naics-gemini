# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
import operator
import time
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    import torch

import httpx
import polars as pl

from naics_embedder.utils.config import DirConfig

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Make directories
# -------------------------------------------------------------------------------------------------

def make_directories(dir_config: DirConfig = DirConfig()) -> None:

    logging.info('Directory setup:')
    for dir, dir_path in dir_config.model_dump().items():
        path = Path(dir_path)

        if path.exists():
            logging.info(f'  • {dir} directory already exists: {dir_path}')

        else:
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f'  • Created {dir} directory: {dir_path}')
    
    logging.info('\n')


# -------------------------------------------------------------------------------------------------
# Get relationship mapping
# -------------------------------------------------------------------------------------------------

def map_relationships(
    key: Union[str, int]
) -> Union[Dict[str, int], Dict[int, str]]:

    relation_map = [
        ('child', 1),
        ('sibling', 2),
        ('grandchild', 3),
        ('great-grandchild', 4),
        ('nephew/niece', 5),
        ('great-great-grandchild', 6),
        ('cousin', 7),
        ('grand-nephew/niece', 8),
        ('grand-grand-nephew/niece', 9),
        ('cousin_1_times_removed', 10),
        ('second_cousin', 11),
        ('cousin_2_times_removed', 12),
        ('second_cousin_1_times_removed', 13),
        ('third_cousin', 14),
        ('unrelated', 15)
    ]

    if isinstance(key, str):
        return {rel: rel_id for rel, rel_id in relation_map}
    elif isinstance(key, int):
        return {rel_id: rel for rel, rel_id in relation_map}
    else:
        raise ValueError('Key must be either str or int.')
    

# -------------------------------------------------------------------------------------------------
# Get relationship
# -------------------------------------------------------------------------------------------------

def get_relationship(idx_code_i: Union[str, int], idx_code_j: Union[str, int]) -> str:

    filter_list = []
    if isinstance(idx_code_i, str):
        filter_list.append(pl.col('code_i').eq(idx_code_i))
    else:
        filter_list.append(pl.col('idx_i').eq(idx_code_i))

    if isinstance(idx_code_j, str):
        filter_list.append(pl.col('code_j').eq(idx_code_j))
    else:
        filter_list.append(pl.col('idx_j').eq(idx_code_j))

    filters = reduce(operator.and_, filter_list)

    return (
        pl.read_parquet(
            './data/naics_relations.parquet'
        )
        .filter(filters)
        .select('relation')
        .get_column('relation')
        .item()
    )

# -------------------------------------------------------------------------------------------------
# Get distance
# -------------------------------------------------------------------------------------------------

def get_distance(idx_code_i: Union[str, int], idx_code_j: Union[str, int]) -> float:

    filter_list = []
    if isinstance(idx_code_i, str):
        filter_list.append(pl.col('code_i').eq(idx_code_i))
    else:
        filter_list.append(pl.col('idx_i').eq(idx_code_i))

    if isinstance(idx_code_j, str):
        filter_list.append(pl.col('code_j').eq(idx_code_j))
    else:
        filter_list.append(pl.col('idx_j').eq(idx_code_j))

    filters = reduce(operator.and_, filter_list)

    return (
        pl.read_parquet(
            './data/naics_distances.parquet'
        )
        .filter(filters)
        .select('distance')
        .get_column('distance')
        .item()
    )


# -------------------------------------------------------------------------------------------------
# Indices, codes, and mappings
# -------------------------------------------------------------------------------------------------

def get_indices_codes(
    return_type: Literal['codes', 'indices', 'code_to_idx', 'idx_to_code']
) -> Union[List[str], List[int], Dict[str, int], Dict[int, str]]:

    '''
    Extract indices and NAICS codes from a parquet file.
    
    Args:
        return_type: One of 'codes', 'indices', 'code_to_idx', 'idx_to_code'.
        
    Returns:
        One of the following based on return_type:
            codes (List[str]): List of unique NAICS codes.
            indices (List[int]): List of indices for the NAICS codes.
            code_to_idx (Dict[str, int]): Mapping from NAICS codes to indices.
            idx_to_code (Dict[int, str]): Mapping from indices to NAICS codes.
    '''

    idx_code_iter = (
        pl
        .read_parquet(
            './data/naics_descriptions.parquet'
        )
        .select('index', 'code')
        .iter_rows(named=True)
    )

    indices, codes, code_to_idx, idx_to_code = [], [], {}, {}
    for row in idx_code_iter:
        idx, code = row['index'], row['code']
        
        codes.append(code)
        indices.append(idx)
        code_to_idx[code] = idx
        idx_to_code[idx] = code

    match return_type:
        case 'codes':
            return codes
        case 'indices':
            return indices
        case 'code_to_idx':
            return code_to_idx
        case 'idx_to_code':
            return idx_to_code
        case _:
            raise ValueError(
                f'Invalid return_type: {return_type}. '
                'Expected one of: codes, indices, code_to_idx, idx_to_code.'
            )


# -------------------------------------------------------------------------------------------------
# Download with exponential backoff retry
# -------------------------------------------------------------------------------------------------

def download_with_retry(
    url: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0
) -> Optional[bytes]:
    
    '''
    Download content from URL with exponential backoff retry logic.
    
    Returns:
        bytes: The downloaded content
        
    Raises:
        httpx.HTTPError, httpx.TimeoutException, ValueError: If all retries fail
    '''

    last_exception = None
    
    for attempt in range(max_retries + 1):

        try:
            
            resp = httpx.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.content
            
            if not data:
                raise ValueError(f'Empty response received from {url}')
            
            return data
            
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as e:

            last_exception = e
            
            if attempt < max_retries:

                delay = initial_delay * (backoff_factor ** attempt)

                print(f'Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {str(e)}')
                print(f'Retrying in {delay:.1f} seconds...')
                
                time.sleep(delay)

            else:

                print(f'All {max_retries + 1} attempts failed for {url}')

                raise last_exception


# -------------------------------------------------------------------------------------------------
# Parquet stats
# -------------------------------------------------------------------------------------------------

def parquet_stats(
    parquet_df: pl.DataFrame,
    message: str,
    output_parquet: str,
    logger: logging.Logger
) -> None:

    logger.info(f'\nParquet observations: {parquet_df.height: ,}\n')

    schema = (
        parquet_df
        .schema
    )

    rows = [(n, d) for n, d in zip(schema.names(), schema.dtypes())]

    logger.info('Parquet schema: Schema([')
    for name, dtype in rows:
        logger.info(f"    ('{name}', {dtype}),")
    logger.info('])\n')

    logger.info(f'{parquet_df.height: ,} {message}:')
    logger.info(f'  {output_parquet}\n')


# -------------------------------------------------------------------------------------------------
# Device and directory utilities
# -------------------------------------------------------------------------------------------------

def pick_device(device_str: str = 'auto') -> 'torch.device':
    '''
    Pick device for PyTorch operations.
    
    Args:
        device_str: Device string ('auto', 'cuda', 'cpu', 'mps')
    
    Returns:
        torch.device object
    '''
    import torch
    
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_str)


def setup_directory(dir_path: str) -> Path:
    '''
    Setup directory, creating it if it doesn't exist.
    
    Args:
        dir_path: Path to directory
    
    Returns:
        Path object to the directory
    '''
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path