'''
Unit tests for MultiChannelEncoder with LoRA adaptation.

Tests cover:
- Model initialization and configuration
- Multi-channel forward pass
- LoRA parameter efficiency
- MoE fusion and gating
- Hyperbolic projection
- Gradient checkpointing
- Output shapes and manifold validity
'''

import logging

import pytest
import torch
import torch.nn as nn

from naics_embedder.text_model.encoder import MultiChannelEncoder
from naics_embedder.text_model.hyperbolic import check_lorentz_manifold_validity

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def encoder_config():
    '''Minimal encoder configuration for fast testing.'''

    return {
        'base_model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # Smaller model for testing
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'num_experts': 4,
        'top_k': 2,
        'moe_hidden_dim': 512,
        'use_gradient_checkpointing': False,  # Disabled for faster testing
        'curvature': 1.0,
    }

@pytest.fixture
def encoder(encoder_config, test_device):
    '''Create MultiChannelEncoder instance for testing.'''

    model = MultiChannelEncoder(**encoder_config)
    model.to(test_device)
    model.eval()
    return model

@pytest.fixture
def sample_tokenized_inputs(test_device, batch_size=4):
    '''Create sample tokenized inputs for all 4 channels.'''

    seq_length = 32
    channels = ['title', 'description', 'excluded', 'examples']

    inputs = {}
    for channel in channels:
        inputs[channel] = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=test_device),
            'attention_mask': torch.ones(batch_size, seq_length, device=test_device),
        }

    return inputs

@pytest.fixture
def sample_tokenized_inputs_variable_length(test_device, batch_size=4):
    '''Create sample tokenized inputs with variable sequence lengths.'''

    max_seq_length = 32
    channels = ['title', 'description', 'excluded', 'examples']

    inputs = {}
    for channel in channels:
        # Create variable length sequences
        input_ids = torch.randint(0, 1000, (batch_size, max_seq_length), device=test_device)
        attention_mask = torch.ones(batch_size, max_seq_length, device=test_device)

        # Mask out some tokens to simulate variable length
        for i in range(batch_size):
            seq_len = torch.randint(max_seq_length // 2, max_seq_length, (1, )).item()
            attention_mask[i, seq_len:] = 0

        inputs[channel] = {'input_ids': input_ids, 'attention_mask': attention_mask}

    return inputs

# -------------------------------------------------------------------------------------------------
# Test: Initialization
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEncoderInitialization:
    '''Test MultiChannelEncoder initialization and configuration.'''

    def test_encoder_creation(self, encoder, encoder_config):
        '''Test that encoder is created successfully with correct architecture.'''

        assert isinstance(encoder, nn.Module)
        assert encoder.channels == ['title', 'description', 'excluded', 'examples']
        assert len(encoder.encoders) == 4  # type: ignore[arg-type]
        assert encoder.embedding_dim == 384  # all-MiniLM-L6-v2 hidden size
        assert encoder.curvature == encoder_config['curvature']

    def test_lora_adapters_applied(self, encoder):
        '''Test that LoRA adapters are applied to all channel encoders.'''

        for channel in encoder.channels:
            channel_encoder = encoder.encoders[channel]
            # Check that PEFT model has the expected structure
            assert hasattr(channel_encoder, 'base_model')
            assert hasattr(channel_encoder, 'peft_config')

    def test_moe_configuration(self, encoder, encoder_config):
        '''Test MoE module configuration.'''

        assert hasattr(encoder, 'moe')
        assert encoder.moe.num_experts == encoder_config['num_experts']
        assert encoder.moe.top_k == encoder_config['top_k']

        # Check MoE projection layer
        assert hasattr(encoder, 'moe_projection')
        assert isinstance(encoder.moe_projection, nn.Linear)
        # Input: concatenated 4 channels, Output: single embedding_dim
        assert encoder.moe_projection.in_features == encoder.embedding_dim * 4
        assert encoder.moe_projection.out_features == encoder.embedding_dim

    def test_hyperbolic_projection(self, encoder):
        '''Test hyperbolic projection layer exists and is configured.'''

        assert hasattr(encoder, 'hyperbolic_proj')
        assert encoder.hyperbolic_proj.input_dim == encoder.embedding_dim
        assert encoder.hyperbolic_proj.c == encoder.curvature

    def test_trainable_parameters(self, encoder):
        '''Test that LoRA reduces trainable parameters significantly.'''

        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in encoder.parameters())

        # With LoRA, trainable params should be much less than total
        trainable_ratio = trainable_params / total_params
        assert 0.0 < trainable_ratio < 0.5, (
            f'LoRA should reduce trainable params to < 50%, got {trainable_ratio:.2%}'
        )

        logger.info(f'Trainable: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2%})')

    def test_different_curvatures(self, encoder_config, test_device):
        '''Test encoder initialization with different curvature values.'''

        for curvature in [0.1, 1.0, 5.0]:
            encoder_config['curvature'] = curvature
            encoder = MultiChannelEncoder(**encoder_config).to(test_device)
            assert encoder.curvature == curvature
            assert encoder.hyperbolic_proj.c == curvature

# -------------------------------------------------------------------------------------------------
# Test: Forward Pass
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEncoderForwardPass:
    '''Test encoder forward pass with various inputs.'''

    def test_forward_basic(self, encoder, sample_tokenized_inputs):
        '''Test basic forward pass produces correct output structure.'''

        with torch.no_grad():
            output = encoder(sample_tokenized_inputs)

        # Check output keys
        assert 'embedding' in output
        assert 'embedding_euc' in output
        assert 'gate_probs' in output
        assert 'top_k_indices' in output

        # Check that all outputs are tensors
        assert isinstance(output['embedding'], torch.Tensor)
        assert isinstance(output['embedding_euc'], torch.Tensor)
        assert isinstance(output['gate_probs'], torch.Tensor)
        assert isinstance(output['top_k_indices'], torch.Tensor)

    def test_output_shapes(self, encoder, sample_tokenized_inputs):
        '''Test output tensor shapes are correct.'''

        batch_size = sample_tokenized_inputs['title']['input_ids'].shape[0]

        with torch.no_grad():
            output = encoder(sample_tokenized_inputs)

        # Hyperbolic embedding: (batch_size, embedding_dim + 1) for Lorentz model
        assert output['embedding'].shape == (batch_size, encoder.embedding_dim + 1)

        # Euclidean embedding: (batch_size, embedding_dim)
        assert output['embedding_euc'].shape == (batch_size, encoder.embedding_dim)

        # Gate probabilities: (batch_size, num_experts)
        assert output['gate_probs'].shape == (batch_size, encoder.moe.num_experts)

        # Top-k indices: (batch_size, top_k)
        assert output['top_k_indices'].shape == (batch_size, encoder.moe.top_k)

    def test_hyperbolic_manifold_validity(self, encoder, sample_tokenized_inputs):
        '''Test that hyperbolic embeddings lie on the Lorentz manifold.'''

        with torch.no_grad():
            output = encoder(sample_tokenized_inputs)

        embedding_hyp = output['embedding']
        is_valid, _, _ = check_lorentz_manifold_validity(
            embedding_hyp, curvature=encoder.curvature, tolerance=1e-4
        )

        assert is_valid, 'Hyperbolic embeddings not on Lorentz manifold'

    def test_variable_length_sequences(self, encoder, sample_tokenized_inputs_variable_length):
        '''Test forward pass with variable length sequences (using attention masks).'''

        with torch.no_grad():
            output = encoder(sample_tokenized_inputs_variable_length)

        # Should still produce valid outputs
        assert (
            output['embedding'].shape[0] == (
                sample_tokenized_inputs_variable_length['title']['input_ids'].shape[0]
            )
        )
        assert not torch.isnan(output['embedding']).any()
        assert not torch.isinf(output['embedding']).any()

    def test_gate_probabilities_sum_to_one(self, encoder, sample_tokenized_inputs):
        '''Test that MoE gate probabilities sum to 1 for each sample.'''

        with torch.no_grad():
            output = encoder(sample_tokenized_inputs)

        gate_probs = output['gate_probs']
        prob_sums = gate_probs.sum(dim=1)

        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), rtol=1e-5, atol=1e-5)

    def test_top_k_indices_valid(self, encoder, sample_tokenized_inputs):
        '''Test that top-k expert indices are valid and unique per sample.'''

        with torch.no_grad():
            output = encoder(sample_tokenized_inputs)

        top_k_indices = output['top_k_indices']

        # Indices should be in valid range [0, num_experts)
        assert (top_k_indices >= 0).all()
        assert (top_k_indices < encoder.moe.num_experts).all()

        # Each row should have unique indices (no duplicate experts)
        for i in range(top_k_indices.shape[0]):
            indices = top_k_indices[i].tolist()
            assert len(indices) == len(set(indices)), f'Duplicate experts in row {i}: {indices}'

    def test_gradient_flow(self, encoder, sample_tokenized_inputs):
        '''Test that gradients flow through the entire encoder.'''

        encoder.train()
        output = encoder(sample_tokenized_inputs)

        # Compute a dummy loss and backprop
        loss = output['embedding'].sum()
        loss.backward()

        # Check that gradients exist for key parameters
        has_grad = False
        for name, param in encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break

        assert has_grad, 'No gradients computed during backprop'

        encoder.eval()

    def test_batch_size_one(self, encoder, test_device):
        '''Test encoder handles batch size of 1.'''

        single_sample_inputs = {}
        for channel in encoder.channels:
            single_sample_inputs[channel] = {
                'input_ids': torch.randint(0, 1000, (1, 32), device=test_device),
                'attention_mask': torch.ones(1, 32, device=test_device),
            }

        with torch.no_grad():
            output = encoder(single_sample_inputs)

        assert output['embedding'].shape[0] == 1
        assert not torch.isnan(output['embedding']).any()

    def test_large_batch_size(self, encoder, test_device):
        '''Test encoder handles larger batch sizes.'''

        large_batch_size = 32
        large_batch_inputs = {}
        for channel in encoder.channels:
            large_batch_inputs[channel] = {
                'input_ids': torch.randint(0, 1000, (large_batch_size, 32), device=test_device),
                'attention_mask': torch.ones(large_batch_size, 32, device=test_device),
            }

        with torch.no_grad():
            output = encoder(large_batch_inputs)

        assert output['embedding'].shape[0] == large_batch_size

# -------------------------------------------------------------------------------------------------
# Test: Mean Pooling
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestMeanPooling:
    '''Test attention-mask-based mean pooling.'''

    def test_padding_ignored(self, encoder, test_device):
        '''Test that padding tokens are properly masked during mean pooling.'''

        # Create input with half padding
        batch_size = 4
        seq_length = 32
        valid_length = seq_length // 2

        inputs = {}
        for channel in encoder.channels:
            input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=test_device)
            attention_mask = torch.zeros(batch_size, seq_length, device=test_device)
            attention_mask[:, :valid_length] = 1  # Only first half is valid

            inputs[channel] = {'input_ids': input_ids, 'attention_mask': attention_mask}

        with torch.no_grad():
            output = encoder(inputs)

        # Should produce valid embeddings (no NaN or Inf)
        assert not torch.isnan(output['embedding']).any()
        assert not torch.isinf(output['embedding']).any()

    def test_different_lengths_per_sample(self, encoder, test_device):
        '''Test that different sequence lengths per sample are handled correctly.'''

        batch_size = 4
        seq_length = 32

        inputs = {}
        for channel in encoder.channels:
            input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=test_device)
            attention_mask = torch.zeros(batch_size, seq_length, device=test_device)

            # Each sample has different valid length
            for i in range(batch_size):
                valid_len = seq_length // (i + 1)
                attention_mask[i, :valid_len] = 1

            inputs[channel] = {'input_ids': input_ids, 'attention_mask': attention_mask}

        with torch.no_grad():
            output = encoder(inputs)

        # All samples should produce different embeddings
        embeddings = output['embedding_euc']
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                # Embeddings should be different (not exactly equal)
                assert not torch.allclose(embeddings[i], embeddings[j], atol=1e-6)

# -------------------------------------------------------------------------------------------------
# Test: Gradient Checkpointing
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestGradientCheckpointing:
    '''Test gradient checkpointing functionality.'''

    def test_gradient_checkpointing_enabled(self, encoder_config, test_device):
        '''Test encoder with gradient checkpointing enabled.'''

        encoder_config['use_gradient_checkpointing'] = True
        encoder = MultiChannelEncoder(**encoder_config).to(test_device)

        # Check that gradient checkpointing is enabled (implicitly through training)
        encoder.train()

        inputs = {}
        for channel in encoder.channels:
            inputs[channel] = {
                'input_ids': torch.randint(0, 1000, (4, 32), device=test_device),
                'attention_mask': torch.ones(4, 32, device=test_device),
            }

        # Should work with gradient checkpointing
        output = encoder(inputs)
        loss = output['embedding'].sum()
        loss.backward()

        # Gradients should still flow
        has_grad = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
        assert has_grad

# -------------------------------------------------------------------------------------------------
# Test: Numerical Stability
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNumericalStability:
    '''Test encoder numerical stability.'''

    def test_no_nan_or_inf(self, encoder, sample_tokenized_inputs):
        '''Test that forward pass produces no NaN or Inf values.'''

        with torch.no_grad():
            output = encoder(sample_tokenized_inputs)

        for key, tensor in output.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch.isnan(tensor).any(), f'{key} contains NaN'
                assert not torch.isinf(tensor).any(), f'{key} contains Inf'

    def test_extreme_curvature_values(self, encoder_config, test_device):
        '''Test encoder with extreme curvature values.'''

        for curvature in [0.01, 100.0]:
            encoder_config['curvature'] = curvature
            encoder = MultiChannelEncoder(**encoder_config).to(test_device)

            inputs = {}
            for channel in encoder.channels:
                inputs[channel] = {
                    'input_ids': torch.randint(0, 1000, (4, 32), device=test_device),
                    'attention_mask': torch.ones(4, 32, device=test_device),
                }

            with torch.no_grad():
                output = encoder(inputs)

            # Should still produce valid outputs
            assert not torch.isnan(output['embedding']).any()
            assert not torch.isinf(output['embedding']).any()

# -------------------------------------------------------------------------------------------------
# Test: Channel Independence
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestChannelIndependence:
    '''Test that channels are encoded independently.'''

    def test_different_inputs_per_channel(self, encoder, test_device):
        '''Test that different channel inputs produce different embeddings.'''

        batch_size = 4

        # Create two different input sets
        inputs_1 = {}
        inputs_2 = {}

        for channel in encoder.channels:
            inputs_1[channel] = {
                'input_ids': torch.randint(0, 1000, (batch_size, 32), device=test_device),
                'attention_mask': torch.ones(batch_size, 32, device=test_device),
            }
            inputs_2[channel] = {
                'input_ids': torch.randint(0, 1000, (batch_size, 32), device=test_device),
                'attention_mask': torch.ones(batch_size, 32, device=test_device),
            }

        with torch.no_grad():
            output_1 = encoder(inputs_1)
            output_2 = encoder(inputs_2)

        # Outputs should be different
        assert not torch.allclose(output_1['embedding'], output_2['embedding'], atol=1e-4)

    def test_single_channel_change(self, encoder, test_device):
        '''Test that changing only one channel affects the output.'''

        batch_size = 4

        # Create base inputs
        base_inputs = {}
        for channel in encoder.channels:
            base_inputs[channel] = {
                'input_ids': torch.randint(0, 1000, (batch_size, 32), device=test_device),
                'attention_mask': torch.ones(batch_size, 32, device=test_device),
            }

        # Create modified inputs (only change title channel)
        modified_inputs = {k: v for k, v in base_inputs.items()}
        modified_inputs['title'] = {
            'input_ids': torch.randint(0, 1000, (batch_size, 32), device=test_device),
            'attention_mask': torch.ones(batch_size, 32, device=test_device),
        }

        with torch.no_grad():
            output_base = encoder(base_inputs)
            output_modified = encoder(modified_inputs)

        # Outputs should be different
        assert not torch.allclose(output_base['embedding'], output_modified['embedding'], atol=1e-4)
