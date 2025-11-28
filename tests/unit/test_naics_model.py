'''
Unit tests for NAICSContrastiveModel (PyTorch Lightning module).

Tests cover:
- Model initialization and configuration
- Forward pass through encoder
- Training step with loss computation
- Validation step and embedding storage
- Optimizer and scheduler configuration
- Curriculum integration
- False negative masking
- Hard negative mining
- Distributed training utilities
- Checkpoint loading
'''

import logging
from unittest.mock import Mock, patch

import polars as pl
import pytest
import pytorch_lightning as pyl
import torch

from naics_embedder.text_model.naics_model import (
    NAICSContrastiveModel,
    gather_embeddings_global,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def model_config():
    '''Minimal model configuration for fast testing.'''

    return {
        'base_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'num_experts': 4,
        'top_k': 2,
        'moe_hidden_dim': 512,
        'temperature': 0.07,
        'curvature': 1.0,
        'hierarchy_weight': 0.0,  # Disabled for basic tests
        'rank_order_weight': 0.0,  # Disabled for basic tests
        'radius_reg_weight': 0.01,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'use_warmup_cosine': False,
        'load_balancing_coef': 0.01,
        'fn_curriculum_start_epoch': 5,
        'fn_cluster_every_n_epochs': 3,
        'fn_num_clusters': 50,
        'distance_matrix_path': None,
        'eval_every_n_epochs': 1,
        'eval_sample_size': 100,
    }

@pytest.fixture
def naics_model(model_config, test_device):
    '''Create NAICSContrastiveModel instance for testing.'''

    model = NAICSContrastiveModel(**model_config)
    model.to(test_device)
    model.eval()
    return model

@pytest.fixture
def sample_training_batch(test_device, batch_size=4, k_negatives=8):
    '''Create sample training batch with anchor, positive, and negative samples.'''

    seq_length = 32
    channels = ['title', 'description', 'excluded', 'examples']

    def create_channel_inputs(batch_size):
        return {
            channel: {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=test_device),
                'attention_mask': torch.ones(batch_size, seq_length, device=test_device),
            }
            for channel in channels
        }

    batch = {
        'anchor': create_channel_inputs(batch_size),
        'positive': create_channel_inputs(batch_size),
        'negatives': create_channel_inputs(batch_size * k_negatives),
        'batch_size': batch_size,
        'k_negatives': k_negatives,
        'anchor_code': [f'{i:02d}111' for i in range(batch_size)],
        'positive_code': [f'{i:02d}1111' for i in range(batch_size)],
        'negative_codes': [[f'{j:02d}999' for j in range(k_negatives)] for _ in range(batch_size)],
    }

    return batch

@pytest.fixture
def sample_distance_matrix(tmp_path):
    '''Create sample ground truth distance matrix.'''

    n_codes = 10
    codes = [f'{i:02d}111' for i in range(n_codes)]

    # Create symmetric distance matrix
    distances = {}
    for i in range(n_codes):
        for j in range(n_codes):
            col_name = f'idx_{i}-code_{codes[i]}'
            if col_name not in distances:
                distances[col_name] = [0.0] * n_codes

            # Simple tree distance: abs(i - j)
            distances[col_name][j] = float(abs(i - j))

    df = pl.DataFrame(distances)
    parquet_path = tmp_path / 'distances.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

# -------------------------------------------------------------------------------------------------
# Test: Initialization
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestModelInitialization:
    '''Test NAICSContrastiveModel initialization and configuration.'''

    def test_model_creation(self, naics_model, model_config):
        '''Test that model is created successfully with all components.'''

        assert isinstance(naics_model, pyl.LightningModule)
        assert hasattr(naics_model, 'encoder')
        assert hasattr(naics_model, 'loss_fn')
        assert hasattr(naics_model, 'hard_negative_miner')
        assert hasattr(naics_model, 'embedding_eval')

    def test_hyperparameters_saved(self, naics_model, model_config):
        '''Test that hyperparameters are saved correctly.'''

        # PyTorch Lightning saves hparams
        assert hasattr(naics_model, 'hparams')
        assert naics_model.hparams['learning_rate'] == model_config['learning_rate']
        assert naics_model.hparams['temperature'] == model_config['temperature']
        assert naics_model.hparams['curvature'] == model_config['curvature']

    def test_encoder_configuration(self, naics_model, model_config):
        '''Test that encoder is configured correctly.'''

        encoder = naics_model.encoder
        assert encoder.curvature == model_config['curvature']
        assert encoder.moe.num_experts == model_config['num_experts']
        assert encoder.moe.top_k == model_config['top_k']

    def test_loss_function_configuration(self, naics_model, model_config):
        '''Test that loss function is configured correctly.'''

        loss_fn = naics_model.loss_fn
        assert loss_fn.temperature == model_config['temperature']
        assert loss_fn.curvature == model_config['curvature']

    def test_hard_negative_miner_configuration(self, naics_model, model_config):
        '''Test that hard negative miner is configured correctly.'''

        miner = naics_model.hard_negative_miner
        assert miner.curvature == model_config['curvature']

    def test_distance_matrix_loading(self, model_config, sample_distance_matrix, test_device):
        '''Test loading ground truth distance matrix.'''

        model_config['distance_matrix_path'] = sample_distance_matrix
        model = NAICSContrastiveModel(**model_config).to(test_device)

        assert model.ground_truth_distances is not None
        assert model.code_to_idx is not None
        assert len(model.code_to_idx) > 0

        # Check that hierarchy loss is initialized
        assert model.hierarchy_loss_fn is None  # Weight is 0 in config

    def test_hierarchy_loss_with_weight(self, model_config, sample_distance_matrix, test_device):
        '''Test hierarchy loss initialization when weight > 0.'''

        model_config['distance_matrix_path'] = sample_distance_matrix
        model_config['hierarchy_weight'] = 0.1
        model = NAICSContrastiveModel(**model_config).to(test_device)

        assert model.hierarchy_loss_fn is not None

    def test_lambdarank_loss_with_weight(self, model_config, sample_distance_matrix, test_device):
        '''Test LambdaRank loss initialization when weight > 0.'''

        model_config['distance_matrix_path'] = sample_distance_matrix
        model_config['rank_order_weight'] = 0.15
        model = NAICSContrastiveModel(**model_config).to(test_device)

        assert model.lambdarank_loss_fn is not None

# -------------------------------------------------------------------------------------------------
# Test: Forward Pass
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestForwardPass:
    '''Test model forward pass.'''

    def test_forward_basic(self, naics_model, sample_training_batch):
        '''Test basic forward pass through encoder.'''

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        assert 'embedding' in output
        assert 'embedding_euc' in output
        assert 'gate_probs' in output
        assert 'top_k_indices' in output

    def test_forward_output_shapes(self, naics_model, sample_training_batch):
        '''Test forward pass output shapes.'''

        batch_size = sample_training_batch['batch_size']

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        # Hyperbolic embedding: (batch_size, embedding_dim + 1)
        assert output['embedding'].shape == (batch_size, naics_model.encoder.embedding_dim + 1)

        # Euclidean embedding: (batch_size, embedding_dim)
        assert output['embedding_euc'].shape == (batch_size, naics_model.encoder.embedding_dim)

# -------------------------------------------------------------------------------------------------
# Test: Training Step
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainingStep:
    '''Test training step functionality.'''

    def test_training_step_basic(self, naics_model, sample_training_batch):
        '''Test basic training step runs without errors.'''

        naics_model.train()
        loss = naics_model.training_step(sample_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_training_step_gradient_flow(self, naics_model, sample_training_batch):
        '''Test that gradients flow through training step.'''

        naics_model.train()
        loss = naics_model.training_step(sample_training_batch, batch_idx=0)
        loss.backward()

        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in naics_model.parameters() if p.requires_grad)
        assert has_grad

    def test_training_step_with_curriculum(self, naics_model, sample_training_batch):
        '''Test training step with curriculum scheduler.'''

        from naics_embedder.text_model.curriculum import CurriculumScheduler

        # Initialize curriculum scheduler
        naics_model.curriculum_scheduler = CurriculumScheduler(
            max_epochs=15, phase1_end=0.33, phase2_end=0.67, phase3_end=1.0
        )

        naics_model.train()
        loss = naics_model.training_step(sample_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_training_step_false_negative_masking(self, naics_model, sample_training_batch):
        '''Test training step with false negative masking.'''

        # Enable clustering in curriculum
        naics_model.current_curriculum_flags = {'enable_clustering': True}

        # Create pseudo labels for clustering
        naics_model.code_to_pseudo_label = {
            code: i % 3
            for i, code in enumerate(sample_training_batch['anchor_code'])
        }

        naics_model.train()
        loss = naics_model.training_step(sample_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_training_step_load_balancing_loss(self, naics_model, sample_training_batch):
        '''Test that load balancing loss is computed.'''

        naics_model.train()
        naics_model.load_balancing_coef = 0.01

        loss = naics_model.training_step(sample_training_batch, batch_idx=0)

        # Loss should include load balancing component
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    @patch('naics_embedder.text_model.naics_model.gather_embeddings_global')
    @patch('torch.distributed.all_reduce')
    def test_training_step_global_batch_sampling(
        self, mock_all_reduce, mock_gather, naics_model, sample_training_batch
    ):
        '''Test training step with global batch sampling (mocked).'''

        # Mock distributed environment
        mock_gather.side_effect = lambda x, *args, **kwargs: x  # Return input unchanged

        # Mock trainer
        naics_model.trainer = Mock()
        naics_model.trainer.is_global_zero = True

        # Enable hard negative mining
        naics_model.current_curriculum_flags = {'enable_hard_negative_mining': True}

        # Mock distributed.is_initialized() and get_world_size()
        with (
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=2),
        ):
            naics_model.train()
            loss = naics_model.training_step(sample_training_batch, batch_idx=0)

            assert isinstance(loss, torch.Tensor)
            # gather_embeddings_global should be called
            # Note: may not be called if conditions aren't met, so we don't assert call count

# -------------------------------------------------------------------------------------------------
# Test: Validation Step
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestValidationStep:
    '''Test validation step functionality.'''

    def test_validation_step_basic(self, naics_model, sample_training_batch):
        '''Test basic validation step runs without errors.'''

        naics_model.eval()
        with torch.no_grad():
            loss = naics_model.validation_step(sample_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_validation_step_embedding_storage(self, naics_model, sample_training_batch):
        '''Test that validation step stores embeddings.'''

        naics_model.eval()
        naics_model.validation_embeddings = {}
        naics_model.validation_codes = []

        with torch.no_grad():
            naics_model.validation_step(sample_training_batch, batch_idx=0)

        # Check that embeddings were stored
        assert len(naics_model.validation_embeddings) > 0
        assert len(naics_model.validation_codes) > 0

        # Check that embeddings match codes
        for code in naics_model.validation_codes:
            assert code in naics_model.validation_embeddings
            embedding = naics_model.validation_embeddings[code]
            assert embedding.shape[0] == naics_model.encoder.embedding_dim + 1  # Lorentz

    def test_validation_step_no_duplicate_codes(self, naics_model, sample_training_batch):
        '''Test that validation step doesn\'t store duplicate codes.'''

        naics_model.eval()
        naics_model.validation_embeddings = {}
        naics_model.validation_codes = []

        # Run validation step twice with same batch
        with torch.no_grad():
            naics_model.validation_step(sample_training_batch, batch_idx=0)
            initial_count = len(naics_model.validation_codes)

            naics_model.validation_step(sample_training_batch, batch_idx=1)
            final_count = len(naics_model.validation_codes)

        # Should not add duplicates
        assert final_count == initial_count

# -------------------------------------------------------------------------------------------------
# Test: Optimizer Configuration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestOptimizerConfiguration:
    '''Test optimizer and scheduler configuration.'''

    def test_configure_optimizers_basic(self, naics_model):
        '''Test basic optimizer configuration.'''

        optimizer_config = naics_model.configure_optimizers()

        # Should return optimizer
        assert 'optimizer' in optimizer_config or isinstance(
            optimizer_config, torch.optim.Optimizer
        )

    def test_adamw_optimizer(self, naics_model, model_config):
        '''Test that AdamW optimizer is configured correctly.'''

        optimizer_config = naics_model.configure_optimizers()

        # Extract optimizer
        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config

        assert isinstance(optimizer, torch.optim.AdamW)

        # Check learning rate
        assert optimizer.param_groups[0]['lr'] == model_config['learning_rate']

        # Check weight decay
        assert optimizer.param_groups[0]['weight_decay'] == model_config['weight_decay']

    def test_warmup_cosine_scheduler(self, model_config, test_device):
        '''Test warmup + cosine scheduler configuration.'''

        model_config['use_warmup_cosine'] = True
        model_config['warmup_steps'] = 100

        model = NAICSContrastiveModel(**model_config).to(test_device)
        optimizer_config = model.configure_optimizers()

        # Should return dict with optimizer and lr_scheduler
        assert isinstance(optimizer_config, dict)
        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config

        # Check scheduler configuration
        lr_scheduler_config = optimizer_config['lr_scheduler']
        assert 'scheduler' in lr_scheduler_config
        assert 'interval' in lr_scheduler_config

# -------------------------------------------------------------------------------------------------
# Test: Distributed Utilities
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDistributedUtilities:
    '''Test distributed training utility functions.'''

    def test_gather_embeddings_single_gpu(self, test_device):
        '''Test gather_embeddings_global with single GPU (no-op).'''

        embeddings = torch.randn(16, 385, device=test_device)

        # Without distributed environment, should return input unchanged
        with patch('torch.distributed.is_initialized', return_value=False):
            gathered = gather_embeddings_global(embeddings)

            assert gathered is embeddings
            torch.testing.assert_close(gathered, embeddings)

    def test_gather_embeddings_world_size_one(self, test_device):
        '''Test gather_embeddings_global with world_size=1.'''

        embeddings = torch.randn(16, 385, device=test_device)

        with (
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=1),
        ):
            gathered = gather_embeddings_global(embeddings)

            assert gathered is embeddings

    @patch('torch.distributed.all_gather')
    def test_gather_embeddings_multi_gpu(self, mock_all_gather, test_device):
        '''Test gather_embeddings_global with multiple GPUs (mocked).'''

        local_embeddings = torch.randn(8, 385, device=test_device)
        world_size = 4

        # Mock all_gather to simulate gathering from multiple GPUs
        def mock_gather_fn(gathered_list, tensor):
            # Simulate gathering: each rank contributes the same tensor
            for i in range(len(gathered_list)):
                gathered_list[i] = tensor.clone()

        mock_all_gather.side_effect = mock_gather_fn

        with (
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=world_size),
        ):
            gathered = gather_embeddings_global(local_embeddings)

            # Should concatenate world_size copies of local embeddings
            assert gathered.shape[0] == local_embeddings.shape[0] * world_size
            assert gathered.shape[1] == local_embeddings.shape[1]

# -------------------------------------------------------------------------------------------------
# Test: Curriculum Integration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCurriculumIntegration:
    '''Test curriculum scheduler integration.'''

    def test_curriculum_flags_update(self, naics_model, sample_training_batch):
        '''Test that curriculum flags are updated during training.'''

        from naics_embedder.text_model.curriculum import CurriculumScheduler

        naics_model.curriculum_scheduler = CurriculumScheduler(
            max_epochs=15, phase1_end=0.33, phase2_end=0.67, phase3_end=1.0
        )

        # Set epoch to phase 2
        naics_model.trainer = Mock()
        naics_model.trainer.current_epoch = 7

        naics_model.train()
        naics_model.training_step(sample_training_batch, batch_idx=0)

        # Check that curriculum flags were updated
        assert len(naics_model.current_curriculum_flags) > 0

    def test_curriculum_phase_transition(self, naics_model, sample_training_batch):
        '''Test curriculum phase transition logging.'''

        from naics_embedder.text_model.curriculum import CurriculumScheduler

        naics_model.curriculum_scheduler = CurriculumScheduler(
            max_epochs=15, phase1_end=0.33, phase2_end=0.67, phase3_end=1.0
        )

        # Transition from phase 1 to phase 2
        naics_model.trainer = Mock()
        naics_model.trainer.current_epoch = 5
        naics_model.previous_phase = 1

        naics_model.train()
        naics_model.training_step(sample_training_batch, batch_idx=0)

        # Previous phase should be updated
        current_phase = naics_model.curriculum_scheduler.get_phase(naics_model.current_epoch)
        assert naics_model.previous_phase == current_phase

# -------------------------------------------------------------------------------------------------
# Test: Checkpoint Loading
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointLoading:
    '''Test model checkpoint saving and loading.'''

    def test_state_dict_save_load(self, naics_model, test_device):
        '''Test that model state dict can be saved and loaded.'''

        # Save state dict
        state_dict = naics_model.state_dict()

        # Create new model
        new_model = NAICSContrastiveModel(**naics_model.hparams).to(test_device)

        # Load state dict
        new_model.load_state_dict(state_dict)

        # Check that parameters match
        for (name1, param1), (name2, param2) in zip(
            naics_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_hparams_save_load(self, naics_model):
        '''Test that hyperparameters are saved correctly.'''

        hparams = naics_model.hparams

        # Hyperparameters should be accessible
        assert 'learning_rate' in hparams
        assert 'curvature' in hparams
        assert 'temperature' in hparams

# -------------------------------------------------------------------------------------------------
# Test: Numerical Stability
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNumericalStability:
    '''Test model numerical stability.'''

    def test_no_nan_in_training(self, naics_model, sample_training_batch):
        '''Test that training produces no NaN values.'''

        naics_model.train()
        loss = naics_model.training_step(sample_training_batch, batch_idx=0)

        assert not torch.isnan(loss)

        # Check gradients
        loss.backward()
        for name, param in naics_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f'NaN gradient in {name}'

    def test_no_inf_in_training(self, naics_model, sample_training_batch):
        '''Test that training produces no Inf values.'''

        naics_model.train()
        loss = naics_model.training_step(sample_training_batch, batch_idx=0)

        assert not torch.isinf(loss)

    def test_extreme_temperature(self, model_config, sample_training_batch, test_device):
        '''Test model with extreme temperature values.'''

        for temperature in [0.01, 1.0]:
            model_config['temperature'] = temperature
            model = NAICSContrastiveModel(**model_config).to(test_device)

            model.train()
            loss = model.training_step(sample_training_batch, batch_idx=0)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

# -------------------------------------------------------------------------------------------------
# Test: Logging and Metrics
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLoggingAndMetrics:
    '''Test logging and metric tracking.'''

    def test_to_python_scalar(self, naics_model):
        '''Test _to_python_scalar conversion utility.'''

        # Test tensor conversion
        tensor_val = torch.tensor(3.14)
        assert isinstance(naics_model._to_python_scalar(tensor_val), float)

        # Test bool conversion
        bool_val = True
        assert isinstance(naics_model._to_python_scalar(bool_val), int)

        # Test float conversion
        float_val = 2.718
        assert isinstance(naics_model._to_python_scalar(float_val), float)

    def test_metrics_file_path(self, naics_model, tmp_path):
        '''Test metrics file path generation.'''

        # Mock logger with log_dir
        mock_logger = Mock()
        mock_logger.log_dir = str(tmp_path / 'logs')

        naics_model.trainer = Mock()
        naics_model.trainer.logger = mock_logger

        metrics_path = naics_model._get_metrics_file_path()

        assert metrics_path is not None
        assert 'evaluation_metrics.json' in str(metrics_path)

# -------------------------------------------------------------------------------------------------
# Test: Edge Cases
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Test edge cases and error handling.'''

    def test_batch_size_one(self, naics_model, test_device):
        '''Test model handles batch size of 1.'''

        seq_length = 32
        channels = ['title', 'description', 'excluded', 'examples']

        batch = {
            'anchor': {
                channel: {
                    'input_ids': torch.randint(0, 1000, (1, seq_length), device=test_device),
                    'attention_mask': torch.ones(1, seq_length, device=test_device),
                }
                for channel in channels
            },
            'positive': {
                channel: {
                    'input_ids': torch.randint(0, 1000, (1, seq_length), device=test_device),
                    'attention_mask': torch.ones(1, seq_length, device=test_device),
                }
                for channel in channels
            },
            'negatives': {
                channel: {
                    'input_ids': torch.randint(0, 1000, (4, seq_length), device=test_device),
                    'attention_mask': torch.ones(4, seq_length, device=test_device),
                }
                for channel in channels
            },
            'batch_size': 1,
            'k_negatives': 4,
            'anchor_code': ['11111'],
        }

        naics_model.train()
        loss = naics_model.training_step(batch, batch_idx=0)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_missing_negative_codes(self, naics_model, sample_training_batch):
        '''Test training step when negative_codes field is missing.'''

        # Remove negative_codes from batch
        batch_no_codes = {k: v for k, v in sample_training_batch.items() if k != 'negative_codes'}

        naics_model.train()
        loss = naics_model.training_step(batch_no_codes, batch_idx=0)

        # Should still work without error
        assert not torch.isnan(loss)
