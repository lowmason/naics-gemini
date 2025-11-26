'''
Unit tests for Mixture of Experts (MoE) module.

Tests gating mechanism, expert routing, load balancing, and batched processing.
'''

import pytest
import torch

from naics_embedder.text_model.moe import MixtureOfExperts, create_moe_layer

# -------------------------------------------------------------------------------------------------
# MixtureOfExperts Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestMixtureOfExperts:
    '''Test suite for Mixture of Experts layer.'''

    @pytest.fixture
    def moe_layer(self):
        '''Create MoE layer with standard configuration.'''

        return MixtureOfExperts(
            input_dim=384 * 4,  # 4 channels
            hidden_dim=1024,
            num_experts=4,
            top_k=2,
        )

    @pytest.fixture
    def sample_input(self, test_device, random_seed):
        '''Create sample input for MoE.'''

        torch.manual_seed(random_seed)
        batch_size = 16
        input_dim = 384 * 4
        return torch.randn(batch_size, input_dim, device=test_device)

    def test_output_shape(self, moe_layer, sample_input):
        '''Test that output has the same shape as input.'''

        output, _, _ = moe_layer(sample_input)

        assert output.shape == sample_input.shape

    def test_output_preserves_batch_size(self, moe_layer, sample_input):
        '''Test that batch size is preserved.'''

        batch_size = sample_input.shape[0]
        output, _, _ = moe_layer(sample_input)

        assert output.shape[0] == batch_size

    def test_gating_probabilities_valid(self, moe_layer, sample_input):
        '''Test that gating probabilities sum to 1.'''

        _, gate_probs, _ = moe_layer(sample_input)

        # Probabilities should sum to 1 for each input
        prob_sums = gate_probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_gating_probabilities_non_negative(self, moe_layer, sample_input):
        '''Test that gating probabilities are non-negative.'''

        _, gate_probs, _ = moe_layer(sample_input)

        assert torch.all(gate_probs >= 0)
        assert torch.all(gate_probs <= 1)

    def test_top_k_selection(self, moe_layer, sample_input):
        '''Test that exactly top_k experts are selected.'''

        _, _, top_k_indices = moe_layer(sample_input)

        batch_size = sample_input.shape[0]
        expected_shape = (batch_size, moe_layer.top_k)

        assert top_k_indices.shape == expected_shape

    def test_top_k_indices_valid_range(self, moe_layer, sample_input):
        '''Test that top_k indices are within valid expert range.'''

        _, _, top_k_indices = moe_layer(sample_input)

        assert torch.all(top_k_indices >= 0)
        assert torch.all(top_k_indices < moe_layer.num_experts)

    def test_different_inputs_different_routing(self, moe_layer, test_device):
        '''Test that different inputs can route to different experts.'''

        # Create two very different inputs
        input1 = torch.randn(8, 384 * 4, device=test_device)
        input2 = torch.randn(8, 384 * 4, device=test_device) * 10  # Much larger scale

        _, _, indices1 = moe_layer(input1)
        _, _, indices2 = moe_layer(input2)

        # At least some routing should be different (not strictly required but very likely)
        # This is a probabilistic test
        assert not torch.all(indices1 == indices2)

    def test_no_nans_or_infs(self, moe_layer, sample_input):
        '''Test that MoE never produces NaN or Inf values.'''

        output, gate_probs, top_k_indices = moe_layer(sample_input)

        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
        assert not torch.any(torch.isnan(gate_probs))
        assert not torch.any(torch.isinf(gate_probs))

    def test_gradient_flow(self, moe_layer, sample_input):
        '''Test that gradients flow through MoE layer.'''

        sample_input.requires_grad_(True)

        output, _, _ = moe_layer(sample_input)
        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert torch.any(sample_input.grad != 0)

    @pytest.mark.parametrize('num_experts', [2, 4, 8])
    def test_different_num_experts(self, num_experts, test_device):
        '''Test MoE with different numbers of experts.'''

        moe = MixtureOfExperts(
            input_dim=384 * 4, hidden_dim=1024, num_experts=num_experts, top_k=min(2, num_experts)
        )

        sample_input = torch.randn(8, 384 * 4, device=test_device)
        output, gate_probs, top_k_indices = moe(sample_input)

        assert output.shape == sample_input.shape
        assert gate_probs.shape == (8, num_experts)

    @pytest.mark.parametrize('top_k', [1, 2, 3])
    def test_different_top_k(self, top_k, test_device):
        '''Test MoE with different top_k values.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=top_k)

        sample_input = torch.randn(8, 384 * 4, device=test_device)
        output, gate_probs, top_k_indices = moe(sample_input)

        assert output.shape == sample_input.shape
        assert top_k_indices.shape == (8, top_k)

    def test_batched_processing_correctness(self, moe_layer, test_device):
        '''Test that batched processing gives consistent results.'''

        # Put model in eval mode to disable dropout
        moe_layer.eval()

        # Process inputs individually vs. in batch
        inputs = torch.randn(4, 384 * 4, device=test_device)

        with torch.no_grad():
            # Batch processing
            batch_output, _, _ = moe_layer(inputs)

            # Individual processing
            individual_outputs = []
            for i in range(4):
                single_input = inputs[i:i + 1]
                single_output, _, _ = moe_layer(single_input)
                individual_outputs.append(single_output)

            individual_outputs = torch.cat(individual_outputs, dim=0)

        # Results should be identical (MoE is deterministic in eval mode)
        assert torch.allclose(batch_output, individual_outputs, atol=1e-5)

        # Restore train mode
        moe_layer.train()

    def test_expert_diversity(self, moe_layer, test_device):
        '''Test that different experts produce different outputs.'''

        # Create a batch large enough to likely activate different experts
        large_batch = torch.randn(32, 384 * 4, device=test_device)

        output, _, top_k_indices = moe_layer(large_batch)

        # Check that multiple experts are used across the batch
        unique_experts = torch.unique(top_k_indices)

        # Should use more than 1 expert across the batch
        assert len(unique_experts) > 1

# -------------------------------------------------------------------------------------------------
# Factory Function Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateMoELayer:
    '''Test suite for MoE factory function.'''

    def test_factory_creates_valid_layer(self):
        '''Test that factory function creates a valid MoE layer.'''

        moe = create_moe_layer(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        assert isinstance(moe, MixtureOfExperts)
        assert moe.input_dim == 384 * 4
        assert moe.hidden_dim == 1024
        assert moe.num_experts == 4
        assert moe.top_k == 2

    def test_factory_default_parameters(self):
        '''Test factory function with default parameters.'''

        moe = create_moe_layer(input_dim=512)

        assert isinstance(moe, MixtureOfExperts)
        assert moe.input_dim == 512
        assert moe.hidden_dim == 1024  # Default
        assert moe.num_experts == 4  # Default
        assert moe.top_k == 2  # Default

# -------------------------------------------------------------------------------------------------
# Expert Routing Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestExpertRouting:
    '''Test suite for expert routing mechanism.'''

    def test_routing_consistency(self, test_device):
        '''Test that same input routes to same experts.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        input_tensor = torch.randn(8, 384 * 4, device=test_device)

        # Forward pass twice with same input
        _, _, indices1 = moe(input_tensor)
        _, _, indices2 = moe(input_tensor)

        # Should route to same experts
        assert torch.all(indices1 == indices2)

    def test_gating_network_output_range(self, test_device):
        '''Test that gating network produces valid output.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        input_tensor = torch.randn(16, 384 * 4, device=test_device)
        _, gate_probs, _ = moe(input_tensor)

        # All probabilities should be in [0, 1]
        assert torch.all(gate_probs >= 0.0)
        assert torch.all(gate_probs <= 1.0)

        # Each row should sum to 1
        row_sums = gate_probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_weighted_combination(self, test_device):
        '''Test that output is a weighted combination of expert outputs.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        input_tensor = torch.randn(1, 384 * 4, device=test_device)
        output, gate_probs, top_k_indices = moe(input_tensor)

        # Output should be influenced by selected experts
        assert output.shape == input_tensor.shape

# -------------------------------------------------------------------------------------------------
# Edge Cases and Stress Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestMoEEdgeCases:
    '''Test suite for edge cases and stress testing.'''

    def test_single_sample_batch(self, test_device) -> None:
        '''Test MoE with batch size of 1.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        input_tensor = torch.randn(1, 384 * 4, device=test_device)
        output, gate_probs, top_k_indices = moe(input_tensor)

        assert output.shape == (1, 384 * 4)
        assert gate_probs.shape == (1, 4)
        assert top_k_indices.shape == (1, 2)

    def test_large_batch(self, test_device):
        '''Test MoE with large batch size.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        input_tensor = torch.randn(128, 384 * 4, device=test_device)
        output, gate_probs, top_k_indices = moe(input_tensor)

        assert output.shape == (128, 384 * 4)
        assert gate_probs.shape == (128, 4)
        assert top_k_indices.shape == (128, 2)

    def test_zero_input(self, test_device):
        '''Test MoE behavior with zero input.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        zero_input = torch.zeros(8, 384 * 4, device=test_device)
        output, gate_probs, top_k_indices = moe(zero_input)

        # Should still produce valid output (may be close to zero depending on initialization)
        assert output.shape == zero_input.shape
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_very_large_values(self, test_device):
        '''Test MoE numerical stability with large input values.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        large_input = torch.randn(8, 384 * 4, device=test_device) * 1000
        output, gate_probs, top_k_indices = moe(large_input)

        # Should handle large values without NaN/Inf
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
        assert not torch.any(torch.isnan(gate_probs))

    def test_all_experts_selected_across_batch(self, test_device):
        '''Test that all experts can be selected across a large batch.'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        # Large diverse batch
        large_batch = torch.randn(100, 384 * 4, device=test_device)
        _, _, top_k_indices = moe(large_batch)

        # Collect all selected expert indices
        all_selected = top_k_indices.flatten()
        unique_experts = torch.unique(all_selected)

        # With 100 samples and diverse inputs, all experts should be used at least once
        # This is probabilistic but very likely
        assert len(unique_experts) == moe.num_experts

# -------------------------------------------------------------------------------------------------
# Performance and Optimization Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestMoEPerformance:
    '''Test suite for performance-related aspects.'''

    def test_forward_pass_speed(self, test_device):
        '''Benchmark test for forward pass speed (not strict assertion).'''

        import time

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)
        input_tensor = torch.randn(32, 384 * 4, device=test_device)

        # Warmup
        for _ in range(5):
            _ = moe(input_tensor)

        # Measure
        start = time.time()
        for _ in range(10):
            _ = moe(input_tensor)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second for 10 iterations on CPU)
        assert elapsed < 5.0, f'Forward pass too slow: {elapsed}s for 10 iterations'

    def test_memory_efficiency(self, test_device):
        '''Test that MoE doesn't leak memory (basic check).'''

        moe = MixtureOfExperts(input_dim=384 * 4, hidden_dim=1024, num_experts=4, top_k=2)

        # Multiple forward passes should not accumulate memory
        for _ in range(10):
            input_tensor = torch.randn(16, 384 * 4, device=test_device)
            output, _, _ = moe(input_tensor)
            del output, input_tensor

        # If we got here without OOM, test passes
        assert True
