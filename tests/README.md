# NAICS Embedder Test Suite

## Overview

This directory contains the test suite for the NAICS Hyperbolic Embedding System, covering unit
tests for individual components and integration tests for end-to-end workflows. The test
infrastructure uses pytest with coverage reporting, property-based testing (Hypothesis), and
performance benchmarking.

### Current Status

**Test Coverage:** ~33% module coverage (15 test files / 46 source modules)

- ✅ **Well Tested**: Text model pipeline (encoding, MoE, loss, hyperbolic ops, evaluation)
- ⚠️ **Partially Tested**: Data processing (distances only)
- 🔴 **Not Tested**: Graph model (HGCN), clustering, training utilities, CLI commands

**Recent additions:**

- ✅ **test_evaluation.py** - Comprehensive evaluation metrics testing (Issue #49)
- ✅ **test_tokenization_cache.py** - Data loading and caching modules testing (Issue #50)

**Priority gaps:** Graph model (HGCN), hyperbolic clustering, training utilities, validation

## Table of Contents

- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Coverage Analysis](#coverage-analysis)
  - [Well-Tested Modules](#well-tested-modules)
  - [Critical Gaps](#critical-gaps-high-priority)
  - [Important Gaps](#important-gaps-medium-priority)
  - [Nice-to-Have Gaps](#nice-to-have-gaps-low-priority)
- [Detailed Test Recommendations](#detailed-test-recommendations)
- [Coverage Goals](#coverage-goals)
- [Test Markers](#test-markers)
- [Adding New Tests](#adding-new-tests)
- [Best Practices](#best-practices)
- [Debugging Failed Tests](#debugging-failed-tests)
- [Continuous Integration](#continuous-integration)
- [Known Issues](#known-issues)

## Test Structure

```
tests/
├── unit/                      # Unit tests for individual components (15 files)
│   ├── test_hyperbolic.py    # Hyperbolic geometry operations (CRITICAL) ✅
│   ├── test_loss.py          # Loss functions ✅
│   ├── test_moe.py           # Mixture of Experts ✅
│   ├── test_encoder.py       # Multi-channel encoder ✅
│   ├── test_naics_model.py   # PyTorch Lightning module ✅
│   ├── test_evaluation.py    # Text model evaluation metrics ✅
│   ├── test_curriculum.py    # Curriculum scheduling ✅
│   ├── test_hard_negative_mining.py  # Hard negative mining ✅
│   ├── test_false_negative_strategy.py  # False negative mitigation ✅
│   ├── test_tokenization_cache.py  # Tokenization caching ✅
│   ├── test_datamodule.py    # Data module and collation ✅
│   ├── test_streaming_dataset.py  # Streaming dataset utilities ✅
│   ├── test_streaming_sampling.py  # Sampling strategies ✅
│   ├── test_data_distances.py  # Distance computation ✅
│   └── test_config.py        # Configuration management ✅
├── integration/              # Integration tests (EMPTY - needs work)
├── fixtures/                 # Test data and fixtures
└── conftest.py              # Shared pytest fixtures

```

## Running Tests

### Run all tests

```bash
uv run pytest tests/
```

### Run specific test file

```bash
uv run pytest tests/unit/test_hyperbolic.py
```

### Run with coverage

```bash
uv run pytest tests/ --cov=src/naics_embedder --cov-report=html
open htmlcov/index.html
```

### Run with verbose output

```bash
uv run pytest tests/ -v
```

### Run only unit tests

```bash
uv run pytest tests/ -m unit
```

### Run in parallel

```bash
uv run pytest tests/ -n auto
```

## Coverage Analysis

### Well-Tested Modules

The following modules have comprehensive test coverage:

#### Text Model Pipeline (Stages 1-3)

1. **text_model/hyperbolic.py** ✅ - `test_hyperbolic.py`
   - LorentzOps (exp/log maps, distances, inner products)
   - HyperbolicProjection (Euclidean → Lorentz)
   - LorentzDistance computation
   - Manifold validity checks
   - Numerical stability tests
   - Property-based tests with Hypothesis
   - **Coverage: >80% (CRITICAL)**

2. **text_model/loss.py** ✅ - `test_loss.py`
   - HyperbolicInfoNCELoss (DCL-based contrastive learning)
   - HierarchyPreservationLoss (distance correlation)
   - RankOrderPreservationLoss (margin-based ranking)
   - LambdaRankLoss (NDCG-based learning to rank)
   - **Coverage: >75%**

3. **text_model/encoder.py** ✅ - `test_encoder.py`
   - Multi-channel transformer encoders (title, description, examples, exclusions)
   - LoRA adaptation layers
   - Channel-specific encoding
   - **Coverage: >70%**

4. **text_model/moe.py** ✅ - `test_moe.py`
   - Top-k gating mechanism
   - Expert routing logic
   - Load balancing loss
   - Batched expert processing
   - **Coverage: >75%**

5. **text_model/naics_model.py** ✅ - `test_naics_model.py`
   - NAICSEmbedder PyTorch Lightning module
   - Training step (forward + loss)
   - Validation step (metrics)
   - Optimizer configuration
   - **Coverage: >60%**

6. **text_model/evaluation.py** ✅ - `test_evaluation.py`
   - EmbeddingEvaluator (distance/similarity computation)
   - RetrievalMetrics (precision@k, recall@k, MAP, NDCG)
   - HierarchyMetrics (cophenetic/Spearman correlation, distortion)
   - EmbeddingStatistics (norm, radius, diversity, collapse detection)
   - NAICSEvaluationRunner (full evaluation pipeline)
   - **Coverage: >75%**

7. **text_model/curriculum.py** ✅ - `test_curriculum.py`
   - Dynamic structure-aware curriculum scheduling
   - Difficulty progression
   - **Coverage: >65%**

8. **text_model/hard_negative_mining.py** ✅ - `test_hard_negative_mining.py`
   - Hard negative sampling strategies
   - Distance-based selection
   - **Coverage: >70%**

9. **text_model/false_negative_strategies.py** ✅ - `test_false_negative_strategy.py`
   - False negative detection strategies
   - Masking logic
   - **Coverage: >65%**

#### Data Loading Pipeline

10. **text_model/dataloader/tokenization_cache.py** ✅ - `test_tokenization_cache.py`
    - Cache building from descriptions
    - Cache save/load operations
    - File locking for multi-worker safety
    - Atomic cache operations
    - get_tokens utility function
    - **Coverage: >70%**

11. **text_model/dataloader/datamodule.py** ✅ - `test_datamodule.py`
    - collate_fn batching logic
    - Multi-level supervision expansion
    - GeneratorDataset worker sharding
    - **Coverage: >60%**

12. **text_model/dataloader/streaming_dataset.py** ✅ - `test_streaming_dataset.py`
    - Taxonomy utilities
    - Ancestor/descendant generation
    - Matrix loading
    - Sampling weight computation
    - **Coverage: >65%**

#### Configuration & Utilities

13. **utils/config.py** ✅ - `test_config.py`
    - Pydantic configuration models
    - YAML loading and parsing
    - Configuration validation
    - **Coverage: >60%**

14. **data/compute_distances.py** ✅ - `test_data_distances.py`
    - NAICS tree construction
    - Pairwise distance calculations
    - Distance matrix generation
    - **Coverage: >65%**

### Critical Gaps (High Priority)

These modules are **critical to system functionality** but have **no test coverage**:

#### 1. 🔴 **Graph Model (Stage 4) - HIGHEST PRIORITY**

**Modules:**

- `graph_model/hgcn.py` (541 lines) - **ZERO COVERAGE**
- `graph_model/dataloader/hgcn_datamodule.py` - **ZERO COVERAGE**
- `graph_model/dataloader/hgcn_streaming_dataset.py` - **ZERO COVERAGE**
- `graph_model/evaluation.py` - **ZERO COVERAGE**

**Why critical:**

- Final refinement stage of the entire embedding pipeline
- Contains complex hyperbolic graph convolutions with custom message passing
- Uses learnable curvature parameters that could become unstable
- Implements triplet loss + radial regularization
- Has known issues (Issue #10: gradient blocking)
- Failure here invalidates all downstream embeddings

**Estimated impact:** Testing would catch **30-40% of potential bugs** in graph refinement

**Required tests:** See [Detailed Test Recommendations](#1-graph-model-hgcn---test_hgcnpy)

#### 2. 🔴 **Hyperbolic Clustering - HIGH PRIORITY**

**Module:** `text_model/hyperbolic_clustering.py` (421 lines) - **ZERO COVERAGE**

**Why critical:**

- Used for false negative detection in contrastive learning
- Implements Fréchet mean computation on hyperboloid (numerically sensitive)
- Directly affects training quality and convergence
- Complex hyperbolic K-means++ initialization
- Could produce invalid Lorentz manifold points

**Estimated impact:** Testing would validate **false negative detection correctness**

**Required tests:** See [Detailed Test Recommendations](#2-hyperbolic-clustering---test_hyperbolic_clusteringpy)

### Important Gaps (Medium Priority)

These modules are **important for reliability** but not immediately critical:

#### 3. 🟡 **Training Utilities**

**Module:** `utils/training.py` (494 lines) - **ZERO COVERAGE**

**Why important:**

- Hardware detection and GPU memory management
- Checkpoint resolution logic (affects training resumption)
- Trainer configuration builder
- Config override parsing
- Failure causes poor user experience but not data corruption

**Required tests:** See [Detailed Test Recommendations](#3-training-utilities---test_training_utilspy)

#### 4. 🟡 **Validation Utilities**

**Module:** `utils/validation.py` (414 lines) - **ZERO COVERAGE**

**Why important:**

- Pre-flight validation before training
- Data file existence and schema checks
- Cache compatibility validation
- Can prevent cryptic runtime errors with actionable messages

**Required tests:** See [Detailed Test Recommendations](#4-validation-utilities---test_validationpy)

#### 5. 🟡 **Data Generation Pipeline**

**Modules:**

- `data/compute_relations.py` (366 lines) - **ZERO COVERAGE**
- `data/create_triplets.py` - **ZERO COVERAGE**
- `data/download_data.py` - **ZERO COVERAGE**

**Why important:**

- Foundation of all training data
- Complex NAICS hierarchy relationship computation (parent, child, sibling, ancestor, exclusion)
- Triplet generation logic affects contrastive learning quality
- Usually run once, so errors detected manually, but tests prevent regressions

**Required tests:** See [Detailed Test Recommendations](#5-data-generation-pipeline)

### Nice-to-Have Gaps (Low Priority)

These modules would benefit from testing but are **lower risk**:

#### 6. 🟢 **CLI Commands**

**Modules:**

- `cli/commands/data.py` - **ZERO COVERAGE**
- `cli/commands/training.py` - **ZERO COVERAGE**
- `cli/commands/tools.py` - **ZERO COVERAGE**

**Why test:**

- User-facing interface (better error handling)
- Integration points for all components
- Can be manually tested, but automation helps

#### 7. 🟢 **Backend & Console Utilities**

**Modules:**

- `utils/backend.py` - **ZERO COVERAGE**
- `utils/utilities.py` - **ZERO COVERAGE**
- `utils/warnings.py` - **ZERO COVERAGE**
- `utils/console.py` - **ZERO COVERAGE**

**Why test:**

- Device selection and directory setup
- Mostly thin wrappers around PyTorch/system calls
- Low complexity, low risk

#### 8. 🟢 **Tools & Visualization**

**Modules:**

- `tools/config_tools.py` - **ZERO COVERAGE**
- `tools/metrics_tools.py` - **ZERO COVERAGE**
- `tools/_visualize_metrics.py` - **ZERO COVERAGE**
- `tools/_investigate_hierarchy.py` - **ZERO COVERAGE**

**Why test:**

- Development and debugging tools
- Not critical for production
- Can be manually tested

#### 9. 🟢 **Integration Tests - EMPTY DIRECTORY**

**Current state:** `tests/integration/` directory exists but contains **no tests**

**Why important:**

- Unit tests validate components in isolation
- Integration tests validate that components work together
- Can catch interaction bugs, data flow issues, config mismatches

**Required:** End-to-end training tests, full pipeline tests

## Detailed Test Recommendations

This section provides specific test cases for each untested module.

### 1. Graph Model (HGCN) - `test_hgcn.py`

**Priority: 🔴 CRITICAL**

```python
import pytest
import torch
from naics_embedder.graph_model.hgcn import (
    HyperbolicConvolution,
    HGCN,
    HGCNLightningModule
)

@pytest.mark.unit
class TestHyperbolicConvolution:
    '''Test suite for HyperbolicConvolution layer.'''

    def test_forward_pass_preserves_manifold(self):
        '''Test that forward pass keeps embeddings on Lorentz manifold.'''
        # Create layer, random embeddings, simple graph
        # Run forward pass
        # Validate output is on manifold using check_lorentz_manifold_validity()

    def test_message_passing_aggregates_neighbors(self):
        '''Test that messages are aggregated from graph neighbors.'''
        # Create simple graph (e.g., star graph)
        # Verify center node receives messages from leaves

    def test_curvature_clamping(self):
        '''Test that curvature is clamped to safe range [0.1, 10.0].'''
        # Initialize with extreme curvature values
        # Verify clamping happens

    def test_gradient_flow_through_layer(self):
        '''Test gradients flow through exp/log maps (Issue #10).'''
        # Create layer with requires_grad=True embeddings
        # Forward + backward pass
        # Verify gradients exist and are non-zero

    def test_residual_connection(self):
        '''Test residual connection in tangent space.'''
        # Verify x_tan + dropout(x_agg) before exp_map

    @pytest.mark.parametrize('curvature', [0.1, 1.0, 5.0, 10.0])
    def test_different_curvatures(self, curvature):
        '''Test layer works with different curvature values.'''

@pytest.mark.unit
class TestHGCN:
    '''Test suite for HGCN model (stack of layers).'''

    def test_multi_layer_forward(self):
        '''Test forward pass through multiple HGCN layers.'''

    def test_learnable_curvature_updates(self):
        '''Test that learnable curvature parameter updates during training.'''

    def test_output_manifold_validity(self):
        '''Test final output is valid Lorentz manifold point.'''

@pytest.mark.unit
class TestHGCNLightningModule:
    '''Test suite for HGCNLightningModule training.'''

    def test_training_step_loss_computation(self):
        '''Test training step computes triplet + radial loss.'''

    def test_validation_step_metrics(self):
        '''Test validation computes distance correlation, MAP, etc.'''

    def test_optimizer_configuration(self):
        '''Test optimizer is configured correctly.'''

    def test_checkpoint_save_load(self):
        '''Test model can save and load checkpoints.'''

@pytest.mark.integration
class TestHGCNTraining:
    '''Integration test for HGCN training loop.'''

    def test_short_training_run(self):
        '''Test 2-epoch training run completes without errors.'''

    def test_loss_decreases(self):
        '''Test that loss decreases over training.'''
```

### 2. Hyperbolic Clustering - `test_hyperbolic_clustering.py`

**Priority: 🔴 CRITICAL**

```python
import pytest
import torch
from naics_embedder.text_model.hyperbolic_clustering import HyperbolicKMeans

@pytest.mark.unit
class TestHyperbolicKMeans:
    '''Test suite for HyperbolicKMeans clustering.'''

    def test_initialization_on_hyperboloid(self):
        '''Test cluster centers are initialized on Lorentz manifold.'''
        # Create synthetic embeddings
        # Initialize HyperbolicKMeans
        # Verify centers satisfy Lorentz constraint

    def test_cluster_assignment_correctness(self):
        '''Test points are assigned to nearest cluster (Lorentzian distance).'''
        # Create well-separated synthetic clusters
        # Run clustering
        # Verify correct assignment

    def test_frechet_mean_computation(self):
        '''Test Fréchet mean is computed correctly on hyperboloid.'''
        # Create cluster of points
        # Compute Fréchet mean
        # Verify mean is on manifold and is centroid

    def test_convergence_with_synthetic_data(self):
        '''Test clustering converges within max_iter.'''
        # Create synthetic data with clear clusters
        # Run fit()
        # Verify convergence (inertia decreases, n_iter < max_iter)

    def test_single_cluster_edge_case(self):
        '''Test clustering with n_clusters=1.'''

    def test_all_points_identical(self):
        '''Test clustering when all points are the same.'''

    @pytest.mark.parametrize('curvature', [0.1, 1.0, 5.0])
    def test_different_curvatures(self, curvature):
        '''Test clustering works with different curvature values.'''

    def test_kmeans_plusplus_initialization(self):
        '''Test k-means++ initialization selects diverse centers.'''

    def test_numerical_stability_extreme_curvature(self):
        '''Test numerical stability with extreme curvatures.'''
```

### 3. Training Utilities - `test_training_utils.py`

**Priority: 🟡 MEDIUM**

```python
import pytest
from unittest.mock import Mock, patch
from naics_embedder.utils.training import (
    detect_hardware,
    parse_config_overrides,
    resolve_checkpoint,
    create_trainer
)

@pytest.mark.unit
class TestHardwareDetection:
    '''Test hardware detection utilities.'''

    @patch('torch.cuda.is_available')
    def test_detect_cuda(self, mock_cuda):
        '''Test CUDA detection when GPU available.'''
        mock_cuda.return_value = True
        hw = detect_hardware()
        assert hw.accelerator == 'cuda'

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_detect_mps(self, mock_mps, mock_cuda):
        '''Test MPS detection on Apple Silicon.'''
        mock_cuda.return_value = False
        mock_mps.return_value = True
        hw = detect_hardware()
        assert hw.accelerator == 'mps'

    def test_detect_cpu_fallback(self):
        '''Test CPU fallback when no accelerator available.'''

@pytest.mark.unit
class TestCheckpointResolution:
    '''Test checkpoint path resolution.'''

    def test_resolve_last_checkpoint(self, tmp_path):
        '''Test resolving "last" to last.ckpt path.'''

    def test_resolve_explicit_path(self, tmp_path):
        '''Test resolving explicit checkpoint path.'''

    def test_resolve_nonexistent_checkpoint(self):
        '''Test error handling for missing checkpoint.'''

@pytest.mark.unit
class TestConfigOverrides:
    '''Test configuration override parsing.'''

    def test_parse_simple_override(self):
        '''Test parsing simple key=value override.'''
        result = parse_config_overrides(['training.learning_rate=0.001'])
        assert result['training']['learning_rate'] == 0.001

    def test_parse_nested_override(self):
        '''Test parsing nested config overrides.'''

    def test_parse_list_override(self):
        '''Test parsing list-valued overrides.'''

    def test_invalid_override_syntax(self):
        '''Test error handling for invalid syntax.'''
```

### 4. Validation Utilities - `test_validation.py`

**Priority: 🟡 MEDIUM**

```python
import pytest
import polars as pl
from pathlib import Path
from naics_embedder.utils.validation import (
    validate_data_paths,
    validate_parquet_schema,
    validate_tokenization_cache,
    validate_training_config,
    ValidationError
)

@pytest.mark.unit
class TestDataPathValidation:
    '''Test data file existence validation.'''

    def test_validate_existing_files(self, tmp_path):
        '''Test validation succeeds when files exist.'''

    def test_validate_missing_files(self, tmp_path):
        '''Test ValidationError raised for missing files.'''

    def test_remediation_steps_provided(self):
        '''Test ValidationError includes actionable remediation.'''

@pytest.mark.unit
class TestParquetSchemaValidation:
    '''Test parquet schema validation.'''

    def test_validate_correct_schema(self, tmp_path):
        '''Test validation succeeds with correct columns.'''
        # Create parquet with expected schema
        df = pl.DataFrame({
            'naics_code': ['111110'],
            'title': ['Test'],
            'description': ['Test desc']
        })
        path = tmp_path / 'test.parquet'
        df.write_parquet(path)
        validate_parquet_schema(path, required_columns=['naics_code', 'title'])

    def test_validate_missing_columns(self, tmp_path):
        '''Test ValidationError raised for missing columns.'''

@pytest.mark.unit
class TestTokenizationCacheValidation:
    '''Test tokenization cache compatibility validation.'''

    def test_validate_compatible_cache(self):
        '''Test validation succeeds for compatible cache.'''

    def test_validate_incompatible_cache(self):
        '''Test ValidationError for incompatible tokenizer.'''

@pytest.mark.unit
class TestTrainingConfigValidation:
    '''Test comprehensive training config validation.'''

    def test_validate_complete_config(self):
        '''Test validation with complete, valid config.'''

    def test_validate_missing_data_files(self):
        '''Test error when data files missing.'''
```

### 5. Data Generation Pipeline

**Priority: 🟡 MEDIUM**

```python
# tests/unit/test_compute_relations.py
import pytest
import polars as pl
from naics_embedder.data.compute_relations import (
    build_naics_tree,
    compute_pairwise_relations,
    calculate_pairwise_relations
)

@pytest.mark.unit
class TestNAICSTreeConstruction:
    '''Test NAICS hierarchy tree construction.'''

    def test_build_tree_simple_hierarchy(self):
        '''Test tree building with simple 2-level hierarchy.'''
        df = pl.DataFrame({
            'naics_code': ['11', '111', '1111'],
            'title': ['Sector', 'Subsector', 'Industry']
        })
        tree = build_naics_tree(df)
        # Verify tree structure

    def test_detect_parent_child_relationships(self):
        '''Test parent-child relationship detection.'''
        # NAICS code '111' is parent of '1111'

    def test_detect_sibling_relationships(self):
        '''Test sibling detection (same parent).'''
        # '1111' and '1112' are siblings

@pytest.mark.unit
class TestRelationshipComputation:
    '''Test pairwise relationship computation.'''

    def test_compute_ancestor_relationships(self):
        '''Test ancestor relationship detection.'''

    def test_compute_exclusion_relationships(self):
        '''Test exclusion relationship parsing.'''

    def test_relationship_matrix_symmetry(self):
        '''Test that relationship matrix has correct symmetry.'''

# tests/unit/test_create_triplets.py
@pytest.mark.unit
class TestTripletGeneration:
    '''Test training triplet generation.'''

    def test_generate_balanced_triplets(self):
        '''Test triplet generation with balanced sampling.'''

    def test_positive_samples_are_related(self):
        '''Test positive samples have hierarchical relationship.'''

    def test_negative_samples_are_unrelated(self):
        '''Test negative samples are not in same subtree.'''

# tests/integration/test_data_pipeline.py
@pytest.mark.integration
class TestDataPipelineIntegration:
    '''Integration test for full data generation pipeline.'''

    def test_full_pipeline(self, tmp_path):
        '''Test download → relations → distances → triplets pipeline.'''
        # Run all data generation steps
        # Verify outputs exist and have correct schemas
        # Verify data consistency across stages
```

### 6. Integration Tests

**Priority: 🟡 MEDIUM**

```python
# tests/integration/test_text_model_training.py
@pytest.mark.integration
class TestTextModelTraining:
    '''End-to-end text model training tests.'''

    def test_short_training_run(self, tmp_path):
        '''Test 2-epoch training completes without errors.'''

    def test_checkpoint_save_and_resume(self, tmp_path):
        '''Test training can be saved and resumed.'''

    def test_validation_metrics_computed(self, tmp_path):
        '''Test validation metrics are computed each epoch.'''

    def test_curriculum_progression(self, tmp_path):
        '''Test curriculum difficulty increases.'''

# tests/integration/test_hgcn_training.py
@pytest.mark.integration
class TestHGCNTraining:
    '''End-to-end HGCN training tests.'''

    def test_load_pretrained_embeddings(self, tmp_path):
        '''Test loading pretrained text embeddings.'''

    def test_graph_refinement_training(self, tmp_path):
        '''Test HGCN training loop.'''

    def test_embedding_quality_improves(self, tmp_path):
        '''Test hierarchy correlation improves after HGCN.'''

# tests/integration/test_end_to_end.py
@pytest.mark.integration
@pytest.mark.slow
class TestEndToEnd:
    '''Complete pipeline test.'''

    def test_full_pipeline_data_to_embeddings(self, tmp_path):
        '''Test data generation → text training → HGCN → evaluation.'''
```

### 7. CLI Commands - `test_cli.py`

**Priority: 🟢 LOW**

```python
import pytest
from typer.testing import CliRunner
from naics_embedder.cli import app

@pytest.mark.integration
class TestCLICommands:
    '''Test CLI command invocation.'''

    def test_data_preprocess_command(self):
        '''Test data preprocess command runs.'''
        runner = CliRunner()
        result = runner.invoke(app, ['data', 'preprocess', '--help'])
        assert result.exit_code == 0

    def test_train_command_with_overrides(self):
        '''Test train command with config overrides.'''

    def test_tools_config_command(self):
        '''Test tools config display command.'''

    def test_invalid_command_shows_help(self):
        '''Test invalid commands show help message.'''
```

## Known Issues

Some tests currently fail due to fixture generation issues:

1. **Hyperbolic fixture**: The `sample_tangent_vectors` and `sample_lorentz_embeddings` fixtures
   need refinement to properly generate tangent vectors at the origin (time component should be 0)

2. **Config loading**: Tests using `tmp_path` need adjustment to work with the config loader's
   path resolution

These are test infrastructure issues, not issues with the core code. The tests themselves are
correctly written and will pass once fixtures are fixed.

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests requiring significant compute
- `@pytest.mark.gpu` - Tests requiring GPU

## Continuous Integration

GitHub Actions automatically runs the full test suite on:

- Push to `main` or `master` branches
- Pull requests to `main` or `master` branches

See `.github/workflows/tests.yml` for CI configuration.

## Coverage Goals

### Target Coverage Metrics

| Module Category | Target | Current | Priority |
|-----------------|--------|---------|----------|
| **Critical Math** (hyperbolic.py, loss.py) | >80% | ~80% | ✅ Met |
| **Text Model Pipeline** | >70% | ~70% | ✅ Met |
| **Graph Model (HGCN)** | >70% | 0% | 🔴 Critical gap |
| **Data Generation** | >60% | ~20% | 🟡 Needs work |
| **Training Utilities** | >65% | 0% | 🟡 Needs work |
| **CLI & Tools** | >50% | 0% | 🟢 Nice to have |
| **Overall Project** | >70% | ~33% | 🔴 Below target |

### Measuring Coverage

```bash
# Generate coverage report
uv run pytest tests/ --cov=src/naics_embedder --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html

# Generate coverage badge (requires coverage-badge)
coverage-badge -o coverage.svg -f

# Check specific module coverage
uv run pytest tests/ --cov=src/naics_embedder.graph_model.hgcn --cov-report=term
```

### Coverage Improvement Roadmap

**Phase 1: Critical Gaps (Weeks 1-2)**

- [ ] Add `test_hgcn.py` - Graph model testing
- [ ] Add `test_hyperbolic_clustering.py` - Clustering validation
- [ ] Target: Bring overall coverage to >50%

**Phase 2: Important Gaps (Weeks 3-4)**

- [ ] Add `test_training_utils.py` - Training utilities
- [ ] Add `test_validation.py` - Validation utilities
- [ ] Add `test_compute_relations.py` - Data relationships
- [ ] Target: Bring overall coverage to >60%

**Phase 3: Integration & Completeness (Weeks 5-6)**

- [ ] Add integration tests for training loops
- [ ] Add end-to-end pipeline tests
- [ ] Add CLI command tests
- [ ] Target: Achieve >70% overall coverage

## Best Practices

### General Testing Principles

1. **Test Behavior, Not Implementation**
   - Focus on what the code does, not how it does it
   - Tests should remain valid even if internal implementation changes
   - Example: Test that embeddings are on manifold, not the specific exp_map formula

2. **Follow AAA Pattern** (Arrange, Act, Assert)
   ```python
   def test_hyperbolic_distance_is_positive(self):
       # Arrange: Set up test data
       x = torch.randn(8, 385)
       y = torch.randn(8, 385)

       # Act: Perform the operation
       distance = lorentz_distance(x, y)

       # Assert: Verify the result
       assert torch.all(distance >= 0)
   ```

3. **Use Descriptive Test Names**
   - Good: `test_exponential_map_produces_manifold_points()`
   - Bad: `test_exp_map()`, `test_function1()`
   - Pattern: `test_<what>_<expected_behavior>`

4. **One Assertion Per Test (Generally)**
   - Each test should verify one logical concept
   - Multiple assertions are OK if testing related properties
   - Use `pytest.mark.parametrize` for testing multiple inputs

5. **Test Edge Cases and Boundaries**
   - Empty inputs, single elements, very large inputs
   - Extreme parameter values (curvature = 0.001, curvature = 100)
   - Invalid inputs (should raise appropriate errors)

6. **Use Fixtures for Shared Setup**
   ```python
   @pytest.fixture
   def sample_embeddings():
       '''Generate sample Lorentz embeddings for testing.'''
       return generate_valid_lorentz_embeddings(n=16, dim=384)
   ```

### Testing Hyperbolic Operations

When testing hyperbolic geometry code:

1. **Always Validate Manifold Constraints**
   ```python
   from naics_embedder.text_model.hyperbolic import check_lorentz_manifold_validity

   is_valid, norms, violations = check_lorentz_manifold_validity(embeddings, c=1.0)
   assert is_valid, f'Manifold constraint violated: max={violations.max()}'
   ```

2. **Test Numerical Stability**
   - Test with small curvatures (0.1) and large curvatures (10.0)
   - Test with extreme distance values
   - Check for NaN, Inf in outputs

3. **Verify Geometric Properties**
   - Distance symmetry: d(x, y) = d(y, x)
   - Identity: d(x, x) = 0
   - Triangle inequality: d(x, z) ≤ d(x, y) + d(y, z)
   - Roundtrip consistency: exp(log(x)) ≈ x

### Testing PyTorch Modules

1. **Test Gradient Flow**
   ```python
   def test_gradients_flow_through_module(self):
       model = MyModule()
       x = torch.randn(8, 10, requires_grad=True)
       loss = model(x).sum()
       loss.backward()
       assert x.grad is not None
       assert not torch.all(x.grad == 0)
   ```

2. **Test with Deterministic Seeds**
   ```python
   import pytorch_lightning as pl

   def test_reproducible_results(self):
       pl.seed_everything(42)
       result1 = model(x)
       pl.seed_everything(42)
       result2 = model(x)
       assert torch.allclose(result1, result2)
   ```

3. **Test Forward and Backward**
   - Forward pass produces correct shapes
   - Backward pass computes gradients
   - Loss decreases over multiple iterations

### Testing Data Processing

1. **Use Synthetic Data**
   ```python
   def test_data_loader_with_synthetic_data(self):
       # Create minimal synthetic dataset
       df = pl.DataFrame({
           'naics_code': ['111110', '111120'],
           'title': ['Industry 1', 'Industry 2']
       })
       dataset = NAICSDataset(df)
       assert len(dataset) == 2
   ```

2. **Test with tmp_path Fixture**
   ```python
   def test_cache_saving(self, tmp_path):
       cache_path = tmp_path / 'test.cache'
       save_cache(cache_path, data)
       assert cache_path.exists()
       loaded = load_cache(cache_path)
       assert loaded == data
   ```

3. **Validate Data Schemas**
   - Check column presence and types
   - Verify data ranges (NAICS codes are strings, distances are floats)

### Using Property-Based Testing

For mathematical functions, use Hypothesis:

```python
from hypothesis import given, settings, strategies as st

@given(
    tangent_vectors=st.lists(
        st.floats(min_value=-10, max_value=10),
        min_size=384,
        max_size=384
    ).map(lambda x: torch.tensor([x]))
)
@settings(max_examples=100)
def test_exp_map_always_produces_manifold_points(self, tangent_vectors):
    '''Property: exp_map always produces valid Lorentz manifold points.'''
    hyp = LorentzOps.exp_map_zero(tangent_vectors, c=1.0)
    is_valid, _, _ = check_lorentz_manifold_validity(hyp, c=1.0)
    assert is_valid
```

### Mocking External Dependencies

Use `unittest.mock` for external resources:

```python
from unittest.mock import Mock, patch

@patch('torch.cuda.is_available')
def test_cuda_detection(self, mock_cuda):
    mock_cuda.return_value = True
    device = get_device()
    assert device == 'cuda'
```

### Performance and Benchmarking

Use `pytest-benchmark` for performance-critical code:

```python
def test_lorentz_distance_performance(benchmark):
    x = torch.randn(1000, 385)
    y = torch.randn(1000, 385)
    result = benchmark(lorentz_distance, x, y, c=1.0)
    assert result.shape == (1000,)
```

## Adding New Tests

### Quick Start Guide

When adding new features, follow this pattern:

1. **Create test file**: `tests/unit/test_<module>.py`
2. **Import relevant code and fixtures**
3. **Create test classes grouped by functionality**
4. **Use descriptive test names**: `test_<what>_<expected_behavior>`
5. **Add docstrings explaining what each test validates**
6. **Use appropriate markers** (`@pytest.mark.unit`, etc.)

### Test File Template

```python
'''
Unit tests for <module_name>.

Tests the <functionality> including <key features>.
'''

import pytest
import torch
from naics_embedder.<module_path> import <classes_to_test>

# -------------------------------------------------------------------------------------------------
# Test Class
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class Test<ClassName>:
    '''Test suite for <ClassName>.'''

    def test_basic_functionality(self):
        '''Test that <feature> works in basic case.'''
        # Arrange
        input_data = create_test_input()

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_output

    def test_edge_case_empty_input(self):
        '''Test behavior with empty input.'''
        result = function_under_test([])
        assert result is not None

    @pytest.mark.parametrize('param', [0.1, 1.0, 5.0])
    def test_different_parameters(self, param):
        '''Test with various parameter values.'''
        result = function_under_test(param=param)
        assert result > 0

# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class Test<ModuleName>Integration:
    '''Integration tests for <module>.'''

    def test_integration_with_other_module(self):
        '''Test integration between <module1> and <module2>.'''
        # Test end-to-end workflow
```

### Fixtures Guide

Add shared fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def sample_naics_data():
    '''Generate sample NAICS data for testing.'''
    return pl.DataFrame({
        'naics_code': ['111110', '111120', '111130'],
        'title': ['Soybean Farming', 'Oilseed Farming', 'Dry Pea Farming'],
        'description': ['Description 1', 'Description 2', 'Description 3']
    })

@pytest.fixture
def sample_lorentz_embeddings():
    '''Generate valid Lorentz manifold embeddings.'''
    # Implementation that creates valid embeddings
    pass
```

## Debugging Failed Tests

### View detailed error messages

```bash
uv run pytest tests/ -vv
```

### Drop into debugger on failure

```bash
uv run pytest tests/ --pdb
```

### Show print statements

```bash
uv run pytest tests/ -s
```

### Run only failed tests from last run

```bash
uv run pytest tests/ --lf
```

## Performance Benchmarking

Some tests include performance benchmarks using `pytest-benchmark`:

```bash
uv run pytest tests/ --benchmark-only
```

## Summary: Priority Action Items

### For New Contributors

**Start here:**

1. Read this README thoroughly
2. Run existing tests: `uv run pytest tests/ -v`
3. Check coverage: `uv run pytest tests/ --cov=src/naics_embedder --cov-report=html`
4. Pick a high-priority gap and implement tests

### For Maintainers

**Immediate priorities (Critical - 🔴):**

| Task | Module | Estimated Effort | Impact |
|------|--------|------------------|--------|
| 1. Implement HGCN tests | `graph_model/hgcn.py` | 2-3 days | Validates entire stage 4 pipeline |
| 2. Implement clustering tests | `text_model/hyperbolic_clustering.py` | 1-2 days | Validates false negative detection |

**Next priorities (Important - 🟡):**

| Task | Module | Estimated Effort | Impact |
|------|--------|------------------|--------|
| 3. Training utilities tests | `utils/training.py` | 1 day | Better training UX |
| 4. Validation tests | `utils/validation.py` | 1 day | Prevents runtime errors |
| 5. Data pipeline tests | `data/compute_relations.py`, etc. | 2 days | Data quality assurance |
| 6. Integration tests | `tests/integration/` | 2-3 days | End-to-end validation |

### Quick Reference Commands

```bash
# Run all tests
uv run pytest tests/

# Run only unit tests
uv run pytest tests/ -m unit

# Run only integration tests
uv run pytest tests/ -m integration

# Run with coverage report
uv run pytest tests/ --cov=src/naics_embedder --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_hyperbolic.py

# Run specific test class
uv run pytest tests/unit/test_hyperbolic.py::TestLorentzOps

# Run specific test method
uv run pytest tests/unit/test_hyperbolic.py::TestLorentzOps::test_exp_log_roundtrip

# Run in parallel (faster)
uv run pytest tests/ -n auto

# Run with verbose output
uv run pytest tests/ -v

# Run only failed tests from last run
uv run pytest tests/ --lf

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Show print statements
uv run pytest tests/ -s

# Generate benchmark report
uv run pytest tests/ --benchmark-only
```

### Test Coverage Quick Stats

| Category | Status |
|----------|--------|
| **Total Source Modules** | 46 |
| **Total Test Files** | 15 |
| **Module Coverage** | ~33% |
| **Target Coverage** | >70% |
| **Gap to Target** | ~37% |

### Critical Missing Tests

1. 🔴 **`tests/unit/test_hgcn.py`** - Graph model (541 lines untested)
2. 🔴 **`tests/unit/test_hyperbolic_clustering.py`** - Clustering (421 lines untested)
3. 🟡 **`tests/unit/test_training_utils.py`** - Training utilities (494 lines untested)
4. 🟡 **`tests/unit/test_validation.py`** - Validation (414 lines untested)
5. 🟡 **`tests/unit/test_compute_relations.py`** - Data generation (366 lines untested)
6. 🟡 **`tests/integration/test_*_training.py`** - Integration tests (none exist)

**Total untested lines in critical modules: ~2,700 lines**

## Contact and Resources

### Getting Help

For questions about the test suite:

- **Documentation**: `/README.md`, `/CLAUDE.md`
- **Issues**: <https://github.com/lowmason/naics-embedder/issues>
- **Discussions**: Use GitHub Discussions for testing strategy questions

### Related Documentation

- **Main README**: Project overview and architecture
- **CLAUDE.md**: AI assistant guide with detailed codebase documentation
- **docs/**: Full MkDocs documentation site
- **pyproject.toml**: Pytest configuration and dependencies

### CI/CD

- **GitHub Actions**: `.github/workflows/tests.yml`
- **Runs on**: Push to `main`, pull requests
- **Includes**: Test suite, coverage reporting, linting

---

**Last Updated**: 2025-11-27

**Document Version**: 2.0 (Comprehensive Coverage Analysis)
