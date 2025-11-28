'''
Shared pytest fixtures for NAICS Embedder test suite.

This module provides reusable fixtures for testing hyperbolic embeddings,
data processing, and model components.
'''

import logging

import polars as pl
import pytest
import torch

# -------------------------------------------------------------------------------------------------
# Test Configuration
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session')
def test_device():
    '''Get device for testing (CPU for CI compatibility).'''

    return torch.device('cpu')

@pytest.fixture(scope='session')
def random_seed():
    '''Fixed random seed for reproducibility.'''

    return 42

@pytest.fixture(autouse=True)
def set_random_seeds(random_seed):
    '''Automatically set random seeds before each test.'''

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

# -------------------------------------------------------------------------------------------------
# Hyperbolic Geometry Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_tangent_vectors(test_device, random_seed):
    '''Generate sample tangent vectors for testing exponential map.

    Vectors are normalized to have reasonable norms (~1-2) to avoid numerical
    overflow in sinh/cosh computations during exp_map.
    '''

    torch.manual_seed(random_seed)
    batch_size = 16
    dim = 384
    # Create random tangent vectors with time component = 0 (proper tangent at origin)
    tangent = torch.randn(batch_size, dim + 1, device=test_device)
    tangent[:, 0] = 0.0  # Time component should be 0 for tangent at origin
    # Scale to have reasonable norms for numerical stability
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0
    return tangent

@pytest.fixture
def sample_lorentz_embeddings(test_device, random_seed):
    '''Generate valid Lorentz embeddings for testing.

    Uses scaled tangent vectors to ensure numerical stability in exp_map.
    '''

    from naics_embedder.text_model.hyperbolic import LorentzOps

    torch.manual_seed(random_seed)
    batch_size = 16
    dim = 384
    # Create tangent vectors with reasonable norms for numerical stability
    tangent = torch.randn(batch_size, dim + 1, device=test_device)
    tangent[:, 0] = 0.0  # Time component should be 0 for tangent at origin
    # Scale to have norm ~2 (avoids sinh/cosh overflow)
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0
    return LorentzOps.exp_map_zero(tangent, c=1.0)

@pytest.fixture
def sample_euclidean_embeddings(test_device, random_seed):
    '''Generate sample Euclidean embeddings.'''

    torch.manual_seed(random_seed)
    batch_size = 16
    dim = 384
    return torch.randn(batch_size, dim, device=test_device)

@pytest.fixture(params=[0.1, 0.5, 1.0, 5.0, 10.0])
def curvature_values(request):
    '''Parametrize tests across different curvature values.'''

    return request.param

# -------------------------------------------------------------------------------------------------
# Data Processing Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_naics_data(tmp_path):
    '''Create minimal NAICS data for testing.'''

    data = {
        'index':
        list(range(15)),
        'code': [
            '31',
            '311',
            '3111',
            '31111',
            '311111',  # Manufacturing - Food
            '32',
            '321',
            '3211',
            '32111',
            '321111',  # Manufacturing - Wood
            '44',
            '441',
            '4411',
            '44111',
            '441111',  # Retail - Motor vehicles
        ],
        'level': [2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
        'title': [
            'Manufacturing',
            'Food Manufacturing',
            'Animal Food Manufacturing',
            'Animal Food Manufacturing',
            'Dog and Cat Food Manufacturing',
            'Manufacturing',
            'Wood Product Manufacturing',
            'Sawmills and Wood Preservation',
            'Sawmills and Wood Preservation',
            'Sawmills',
            'Retail Trade',
            'Motor Vehicle and Parts Dealers',
            'Automobile Dealers',
            'New Car Dealers',
            'New Car Dealers',
        ],
    }

    df = pl.DataFrame(data)
    parquet_path = tmp_path / 'naics_test.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

@pytest.fixture
def sample_naics_relations(tmp_path, sample_naics_data):
    '''Create sample NAICS relationship data.'''

    relations_data = {
        'idx_i': [0, 1, 2, 3, 5, 6, 7, 10, 11, 12],
        'idx_j': [1, 2, 3, 4, 6, 7, 8, 11, 12, 13],
        'code_i': ['31', '311', '3111', '31111', '32', '321', '3211', '44', '441', '4411'],
        'code_j': [
            '311',
            '3111',
            '31111',
            '311111',
            '321',
            '3211',
            '32111',
            '441',
            '4411',
            '44111',
        ],
        'relation': ['child'] * 10,
        'relation_id': [1] * 10,
    }

    df = pl.DataFrame(relations_data)
    parquet_path = tmp_path / 'naics_relations_test.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

@pytest.fixture
def sample_naics_distances(tmp_path):
    '''Create sample NAICS distance data.'''

    distances_data = {
        'idx_i': [0, 0, 0, 1, 1, 2],
        'idx_j': [1, 2, 3, 2, 3, 3],
        'code_i': ['31', '31', '31', '311', '311', '3111'],
        'code_j': ['311', '3111', '31111', '3111', '31111', '31111'],
        'distance': [0.5, 1.5, 2.5, 0.5, 1.5, 0.5],
    }

    df = pl.DataFrame(distances_data)
    parquet_path = tmp_path / 'naics_distances_test.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

# -------------------------------------------------------------------------------------------------
# Model Component Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def batch_size():
    '''Standard batch size for testing.'''

    return 16

@pytest.fixture
def embedding_dim():
    '''Standard embedding dimension for testing.'''

    return 384

@pytest.fixture
def num_channels():
    '''Number of text channels (title, description, examples, exclusions).'''

    return 4

@pytest.fixture
def sample_batch(batch_size, num_channels, test_device, random_seed):
    '''Generate sample batch of multi-channel embeddings.'''

    torch.manual_seed(random_seed)
    return {
        'title': torch.randn(batch_size, 384, device=test_device),
        'description': torch.randn(batch_size, 384, device=test_device),
        'examples': torch.randn(batch_size, 384, device=test_device),
        'exclusions': torch.randn(batch_size, 384, device=test_device),
    }

# -------------------------------------------------------------------------------------------------
# Logging Configuration for Tests
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def configure_logging():
    '''Configure logging for test runs.'''

    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(levelname)s - %(name)s - %(message)s',
    )

    # Suppress specific noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

# -------------------------------------------------------------------------------------------------
# Temporary Directory Helpers
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    '''Create temporary checkpoint directory.'''

    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    return checkpoint_dir

@pytest.fixture
def temp_data_dir(tmp_path):
    '''Create temporary data directory.'''

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def temp_config_dir(tmp_path):
    '''Create temporary config directory.'''

    config_dir = tmp_path / 'conf'
    config_dir.mkdir()
    return config_dir
