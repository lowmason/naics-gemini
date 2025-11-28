# Test Coverage Analysis for NAICS Embedder

**Analysis Date:** 2025-11-28
**Test Suite Size:** ~9,093 lines across 18 unit test files
**Test Count:** 471 unit tests
**Source Files:** 54 Python files (41 excluding `__init__.py` and internal tools)

## Executive Summary

The NAICS Embedder project has **58.5% file-level test coverage**, with 24 of 41 major source files
having dedicated test files. While core ML components (encoders, losses, models) have solid test
coverage, critical infrastructure components lack tests entirely.

**Key Findings:**
- ✅ Strong coverage of ML components (encoder, MoE, hyperbolic ops, losses, evaluation)
- ✅ Good coverage of data processing and curriculum learning
- ❌ **Zero coverage** of CLI commands (661 lines untested in `training.py` alone)
- ❌ **Zero coverage** of critical utilities (training, validation, backend, console)
- ❌ **Zero coverage** of HGCN model (1,101 lines untested)
- ❌ **Zero integration tests** (placeholder directory only)

## Detailed Coverage Breakdown

### 1. Well-Tested Components ✅

These modules have dedicated test files with good coverage:

| Module | Test File | Test Focus |
|--------|-----------|------------|
| `text_model/encoder.py` | `test_encoder.py` | Multi-channel LoRA encoding |
| `text_model/moe.py` | `test_moe.py` | Mixture-of-Experts fusion |
| `text_model/loss.py` | `test_loss.py` | DCL loss, hierarchy loss |
| `text_model/hyperbolic.py` | `test_hyperbolic.py` | Lorentz operations |
| `text_model/curriculum.py` | `test_curriculum.py` | Dynamic curriculum |
| `text_model/evaluation.py` | `test_evaluation.py` | Embedding metrics |
| `text_model/hard_negative_mining.py` | `test_hard_negative_mining.py` | Hard negative mining |
| `text_model/false_negative_strategies.py` | `test_false_negative_strategy.py` | FN strategies |
| `text_model/naics_model.py` | `test_naics_model.py` | Main Lightning module |
| `text_model/dataloader/*` | `test_datamodule.py`, `test_streaming_dataset.py`, `test_tokenization_cache.py` | Data loading |
| `data/compute_distances.py` | `test_data_distances.py` | Distance computation |
| `graph_model/curriculum/*` | `test_graph_curriculum.py`, `test_graph_preprocessing.py` | Graph curriculum |
| `utils/config.py` | `test_config.py` | Configuration validation |
| `utils/hyperbolic.py` | `test_utils_hyperbolic.py` | Manifold abstraction |

### 2. Untested Critical Components ❌

These modules have **NO test coverage** despite being critical:

#### **High Priority (Critical Path)**

| Module | Lines | Criticality | Risk |
|--------|-------|-------------|------|
| `graph_model/hgcn.py` | 1,101 | **CRITICAL** | Graph refinement stage untested |
| `cli/commands/training.py` | 661 | **HIGH** | Main training orchestration untested |
| `utils/training.py` | ~250 | **HIGH** | Hardware detection, trainer creation |
| `utils/validation.py` | ~350 | **HIGH** | Pre-flight validation logic |
| `text_model/naics_model.py` | 2,112 | **PARTIAL** | Some coverage, but 2K+ lines needs more |

#### **Medium Priority (Infrastructure)**

| Module | Lines | Criticality | Risk |
|--------|-------|-------------|------|
| `cli/commands/data.py` | ~200 | **MEDIUM** | Data pipeline orchestration |
| `cli/commands/tools.py` | ~150 | **MEDIUM** | Tool commands untested |
| `utils/backend.py` | ~100 | **MEDIUM** | Device detection logic |
| `utils/console.py` | ~150 | **MEDIUM** | Rich console formatting |
| `utils/utilities.py` | ~200 | **MEDIUM** | Helper functions |
| `utils/warnings.py` | ~80 | **LOW** | Warning suppression config |

#### **Medium Priority (Data Processing)**

| Module | Lines | Criticality | Risk |
|--------|-------|-------------|------|
| `data/download_data.py` | ~300 | **MEDIUM** | NAICS data download/processing |
| `data/compute_relations.py` | ~250 | **MEDIUM** | Relation computation |
| `data/create_triplets.py` | ~200 | **MEDIUM** | Training pair generation |

#### **Low Priority (Tools)**

| Module | Lines | Criticality | Risk |
|--------|-------|-------------|------|
| `tools/config_tools.py` | ~100 | **LOW** | Config display utilities |
| `tools/metrics_tools.py` | ~150 | **LOW** | Metrics visualization |

### 3. Integration Test Gap ⚠️

**Current State:** The `tests/integration/` directory contains only an `__init__.py` file.

**Missing Coverage:**
- No end-to-end workflow tests
- No multi-stage pipeline tests (text → graph)
- No checkpoint loading/resumption tests
- No distributed training tests
- No CLI integration tests

## Recommendations

### Priority 1: Critical Components (Immediate Action)

#### 1.1 Test `graph_model/hgcn.py` (1,101 lines)

**Why:** This is the entire Stage 4 of the pipeline and has zero test coverage.

**Test Cases Needed:**
- Hyperbolic convolution forward pass correctness
- Lorentz manifold constraint preservation
- Curriculum phase transitions (4 phases)
- Event bus coordination
- Adaptive loss computation (MACL)
- Phase-specific sampling strategies
- Gradient flow in hyperbolic space
- Edge case handling (zero-degree nodes, disconnected components)

**Suggested Test File:** `tests/unit/test_hgcn.py`

**Example Test Structure:**
```python
class TestHyperbolicConvolution:
    def test_forward_preserves_lorentz_constraint(self):
        '''Verify output embeddings lie on Lorentz manifold'''

    def test_edge_aware_attention_weighting(self):
        '''Verify attention weights sum to 1 and respect edge types'''

    def test_gradient_flow_through_hyperbolic_ops(self):
        '''Ensure gradients flow correctly through exp/log maps'''

class TestHGCNModel:
    def test_curriculum_phase_transitions(self):
        '''Verify correct transitions between 4 curriculum phases'''

    def test_validation_metrics_computation(self):
        '''Test MAP, hierarchy correlation, level consistency'''

    def test_checkpoint_save_load(self):
        '''Verify model state persists correctly'''

class TestAdaptiveLoss:
    def test_macl_margin_adaptation(self):
        '''Verify confidence-based margin adjustment'''

    def test_curriculum_weight_scheduling(self):
        '''Test loss weight evolution across phases'''
```

#### 1.2 Test `utils/training.py` (~250 lines)

**Why:** Core training orchestration utilities used by all training commands.

**Test Cases Needed:**
- Hardware detection (CUDA, MPS, CPU)
- GPU memory info parsing
- Checkpoint resolution logic
- Trainer creation with various configs
- Config override parsing
- Training result dataclass serialization

**Suggested Test File:** `tests/unit/test_utils_training.py`

**Example Test Structure:**
```python
class TestHardwareDetection:
    def test_detect_hardware_cuda(self, mock_cuda):
        '''Mock CUDA available, verify correct detection'''

    def test_detect_hardware_mps(self, mock_mps):
        '''Mock MPS available, verify detection'''

    def test_detect_hardware_cpu_fallback(self):
        '''No GPU available, verify CPU fallback'''

    def test_get_gpu_memory_info(self, mock_gpu):
        '''Verify GPU memory parsing'''

class TestCheckpointResolution:
    def test_resolve_checkpoint_last(self, tmp_path):
        '''Resolve "last" to latest checkpoint'''

    def test_resolve_checkpoint_best(self, tmp_path):
        '''Resolve "best" to lowest val loss'''

    def test_resolve_checkpoint_explicit_path(self):
        '''Use explicit path as-is'''

    def test_resolve_checkpoint_missing_raises(self):
        '''Missing checkpoint raises clear error'''

class TestTrainerCreation:
    def test_create_trainer_default_config(self):
        '''Create trainer with default settings'''

    def test_create_trainer_custom_callbacks(self):
        '''Create trainer with custom callbacks'''

    def test_create_trainer_multi_gpu(self):
        '''Create trainer with DDP strategy'''
```

#### 1.3 Test `utils/validation.py` (~350 lines)

**Why:** Pre-flight validation prevents runtime failures, but validation logic itself is untested.

**Test Cases Needed:**
- Data path validation (missing files, wrong format)
- Parquet schema validation (column presence, types)
- Config validation (field types, ranges, dependencies)
- Tokenization cache validation
- `ValidationError` exception with remediation steps

**Suggested Test File:** `tests/unit/test_utils_validation.py`

**Example Test Structure:**
```python
class TestDataPathValidation:
    def test_validate_data_paths_all_present(self, tmp_path):
        '''All required files exist'''

    def test_validate_data_paths_missing_file(self, tmp_path):
        '''Missing file returns error with remediation'''

    def test_validate_data_paths_wrong_extension(self, tmp_path):
        '''.csv instead of .parquet returns error'''

class TestParquetSchemaValidation:
    def test_validate_descriptions_schema(self, tmp_path):
        '''Verify descriptions parquet has required columns'''

    def test_validate_distances_schema(self, tmp_path):
        '''Verify distances parquet schema'''

    def test_validate_missing_column_raises(self, tmp_path):
        '''Missing required column returns error'''

class TestConfigValidation:
    def test_validate_training_config_valid(self):
        '''Valid config passes validation'''

    def test_validate_invalid_batch_size(self):
        '''Batch size <= 0 returns error'''

    def test_validate_conflicting_settings(self):
        '''Conflicting settings return warning'''

    def test_require_valid_config_raises_on_error(self):
        '''require_valid_config raises ValidationError'''
```

#### 1.4 Test `cli/commands/training.py` (661 lines)

**Why:** Main entry point for training, orchestrates entire pipeline.

**Test Cases Needed:**
- Command-line argument parsing
- Config loading and overrides
- Training workflow orchestration
- Error handling and user feedback
- Checkpoint resumption flow
- Multi-GPU setup

**Suggested Test File:** `tests/unit/test_cli_training.py`

**Example Test Structure:**
```python
class TestTrainingCommand:
    def test_parse_training_args_defaults(self):
        '''Default arguments parsed correctly'''

    def test_parse_training_args_with_overrides(self):
        '''Config overrides applied correctly'''

    def test_training_workflow_validation_failure(self, mock_invalid_data):
        '''Validation failure exits gracefully'''

    def test_training_checkpoint_resume(self, mock_checkpoint):
        '''Resume from checkpoint workflow'''

    def test_training_error_handling(self, mock_training_failure):
        '''Training error reported to user'''
```

### Priority 2: Infrastructure & Data Processing

#### 2.1 Test CLI Commands

**Files:**
- `cli/commands/data.py` - Data preparation pipeline
- `cli/commands/tools.py` - Utility tool commands

**Test File:** `tests/unit/test_cli_commands.py`

**Focus:**
- Command argument parsing
- Error handling and user feedback
- Pipeline orchestration
- Exit codes

#### 2.2 Test Utilities

**Files:**
- `utils/backend.py` - Device detection
- `utils/console.py` - Rich formatting
- `utils/utilities.py` - Helper functions

**Test Files:** `tests/unit/test_utils_backend.py`, `tests/unit/test_utils_console.py`

**Focus:**
- Device detection logic (mock torch.cuda/mps)
- Console table formatting
- Directory creation and cleanup
- Error handling

#### 2.3 Test Data Processing

**Files:**
- `data/download_data.py` - NAICS data download
- `data/compute_relations.py` - Relation metrics
- `data/create_triplets.py` - Training pair generation

**Test Files:** `tests/unit/test_data_download.py`, `tests/unit/test_data_relations.py`,
`tests/unit/test_data_triplets.py`

**Focus:**
- Data transformation correctness
- Edge case handling (empty data, malformed input)
- Output validation (schema, ranges)
- Error handling

### Priority 3: Integration Tests

**Suggested Test Files:** `tests/integration/test_full_pipeline.py`,
`tests/integration/test_checkpoint_workflow.py`

#### 3.1 End-to-End Pipeline Tests

**Test Cases:**
- Full data preparation → text training → HGCN refinement pipeline
- Checkpoint save/load/resume workflow
- Multi-epoch training convergence
- Embedding quality validation

**Example:**
```python
class TestFullPipeline:
    def test_text_to_hgcn_pipeline(self, small_dataset):
        '''Train text model, then refine with HGCN'''
        # 1. Train text model for 2 epochs
        # 2. Generate embeddings
        # 3. Train HGCN for 2 epochs
        # 4. Validate final embeddings
        # 5. Check hierarchy correlation improved

    def test_checkpoint_resume_workflow(self, tmp_path):
        '''Train, interrupt, resume, verify continuity'''
        # 1. Train for 3 epochs
        # 2. Load checkpoint
        # 3. Resume for 2 more epochs
        # 4. Verify epoch count is 5
        # 5. Verify metrics continuity
```

#### 3.2 CLI Integration Tests

**Test Cases:**
- CLI commands execute successfully end-to-end
- Config overrides work correctly
- Error messages are helpful
- Exit codes are correct

**Example:**
```python
class TestCLIIntegration:
    def test_data_all_command(self, tmp_path):
        '''Run naics-embedder data all successfully'''

    def test_train_command_with_config(self, tmp_path):
        '''Run training with custom config'''

    def test_train_command_with_overrides(self, tmp_path):
        '''Run training with CLI overrides'''

    def test_tools_visualize_command(self, tmp_path):
        '''Run visualization tool successfully'''
```

### Priority 4: Expand Existing Test Coverage

#### 4.1 `text_model/naics_model.py` (2,112 lines, partial coverage)

While this file has tests, the 2,112 lines suggest many code paths are untested. Expand tests for:

- All validation metrics computation paths
- Distributed training (multi-GPU) paths
- Edge cases in loss computation
- Curriculum transitions
- False negative handling strategies
- Gradient accumulation

#### 4.2 Add Property-Based Testing

Use `hypothesis` library (already in dependencies) for:

- Hyperbolic operations (verify mathematical properties)
- Distance metrics (symmetry, triangle inequality)
- Loss functions (non-negativity, bounds)
- Sampling strategies (distribution properties)

**Example:**
```python
from hypothesis import given, strategies as st

class TestHyperbolicProperties:
    @given(
        x=st.tensors(dtype=torch.float32, shape=(10, 64)),
        y=st.tensors(dtype=torch.float32, shape=(10, 64))
    )
    def test_distance_symmetry(self, x, y):
        '''d(x, y) == d(y, x)'''
        x_hyp = exp_map_zero(x)
        y_hyp = exp_map_zero(y)
        d_xy = lorentz_distance(x_hyp, y_hyp)
        d_yx = lorentz_distance(y_hyp, x_hyp)
        assert torch.allclose(d_xy, d_yx)

    @given(
        x=st.tensors(dtype=torch.float32, shape=(10, 64)),
        y=st.tensors(dtype=torch.float32, shape=(10, 64)),
        z=st.tensors(dtype=torch.float32, shape=(10, 64))
    )
    def test_triangle_inequality(self, x, y, z):
        '''d(x, z) <= d(x, y) + d(y, z)'''
        x_hyp = exp_map_zero(x)
        y_hyp = exp_map_zero(y)
        z_hyp = exp_map_zero(z)
        d_xz = lorentz_distance(x_hyp, z_hyp)
        d_xy = lorentz_distance(x_hyp, y_hyp)
        d_yz = lorentz_distance(y_hyp, z_hyp)
        assert (d_xz <= d_xy + d_yz + 1e-5).all()  # Small tolerance for numerics
```

## Testing Infrastructure Improvements

### 1. Add Coverage Reporting to CI

**Modify `.github/workflows/tests.yml`:**
```yaml
- name: Run tests with coverage
  run: uv run pytest --cov=naics_embedder --cov-report=xml --cov-report=term

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    fail_ci_if_error: true

- name: Check minimum coverage
  run: |
    uv run pytest --cov=naics_embedder --cov-fail-under=75
```

### 2. Add Coverage Badge to README

```markdown
[![Coverage](https://codecov.io/gh/lowmason/naics-embedder/branch/main/graph/badge.svg)](https://codecov.io/gh/lowmason/naics-embedder)
```

### 3. Set Up Pre-Commit Coverage Checks

Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: pytest-coverage
      name: Check test coverage
      entry: uv run pytest --cov=naics_embedder --cov-fail-under=75 -q
      language: system
      pass_filenames: false
      always_run: true
```

### 4. Create Test Fixtures Directory

Organize shared test data:
```
tests/
├── fixtures/
│   ├── __init__.py
│   ├── sample_data.py       # Sample NAICS data generators
│   ├── mock_configs.py      # Mock configuration objects
│   ├── mock_checkpoints.py  # Mock checkpoint data
│   └── sample_embeddings.py # Pre-computed test embeddings
```

### 5. Add Performance Benchmarks

Use `pytest-benchmark` (already in dependencies):
```python
def test_hyperbolic_distance_performance(benchmark):
    '''Benchmark Lorentz distance computation'''
    x = torch.randn(1000, 128)
    y = torch.randn(1000, 128)
    x_hyp = exp_map_zero(x)
    y_hyp = exp_map_zero(y)

    result = benchmark(lorentz_distance, x_hyp, y_hyp)
    assert result.shape == (1000,)
```

## Summary of Recommended New Test Files

| Priority | Test File | Target Module(s) | Estimated Tests |
|----------|-----------|------------------|-----------------|
| **P1** | `tests/unit/test_hgcn.py` | `graph_model/hgcn.py` | 40-50 |
| **P1** | `tests/unit/test_utils_training.py` | `utils/training.py` | 25-30 |
| **P1** | `tests/unit/test_utils_validation.py` | `utils/validation.py` | 30-35 |
| **P1** | `tests/unit/test_cli_training.py` | `cli/commands/training.py` | 20-25 |
| **P2** | `tests/unit/test_cli_commands.py` | `cli/commands/data.py`, `tools.py` | 15-20 |
| **P2** | `tests/unit/test_utils_backend.py` | `utils/backend.py` | 10-15 |
| **P2** | `tests/unit/test_utils_console.py` | `utils/console.py` | 10-12 |
| **P2** | `tests/unit/test_data_download.py` | `data/download_data.py` | 15-20 |
| **P2** | `tests/unit/test_data_relations.py` | `data/compute_relations.py` | 15-20 |
| **P2** | `tests/unit/test_data_triplets.py` | `data/create_triplets.py` | 15-20 |
| **P3** | `tests/integration/test_full_pipeline.py` | Full pipeline | 8-10 |
| **P3** | `tests/integration/test_checkpoint_workflow.py` | Checkpoint handling | 5-8 |
| **P3** | `tests/integration/test_cli.py` | CLI commands | 10-15 |
| **Total** | **13 new test files** | **17 untested modules** | **218-280 tests** |

## Expected Impact

Implementing these recommendations would:

- **Increase file coverage** from 58.5% to 100% (all major modules tested)
- **Increase line coverage** from ~60% (estimated) to 80%+ (target)
- **Add ~250 new tests**, bringing total to ~720 tests
- **Add integration test suite** (currently non-existent)
- **Reduce production bugs** by catching issues in critical paths (training, validation, CLI)
- **Improve confidence** in refactoring and adding new features
- **Catch regressions** early in CI pipeline

## Next Steps

1. **Immediate (Week 1):**
   - Create `test_hgcn.py` (P1, highest risk)
   - Create `test_utils_training.py` (P1, high usage)
   - Create `test_utils_validation.py` (P1, critical path)

2. **Short-term (Week 2-3):**
   - Create `test_cli_training.py` (P1)
   - Create remaining P2 utility tests
   - Add coverage reporting to CI

3. **Medium-term (Month 1):**
   - Create P2 data processing tests
   - Expand existing test coverage (naics_model.py)
   - Add property-based tests for hyperbolic ops

4. **Long-term (Month 2+):**
   - Build out integration test suite
   - Add performance benchmarks
   - Set coverage targets per module (80%+ goal)
