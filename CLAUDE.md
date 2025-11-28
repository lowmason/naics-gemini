# CLAUDE.md - AI Assistant Guide for NAICS Embedder

## Project Overview

**NAICS Hyperbolic Embedding System** is a sophisticated machine learning framework that produces
hyperbolic embeddings for the North American Industry Classification System (NAICS). The system
combines multi-channel text encoding, Mixture-of-Experts fusion, hyperbolic contrastive learning,
and hyperbolic graph refinement to create geometry-aware embeddings aligned with the hierarchical
NAICS taxonomy.

**Key Technologies:**

- **Language:** Python 3.10+
- **Package Manager:** `uv` (modern Python package manager)
- **ML Frameworks:** PyTorch ≥2.4.0, PyTorch Lightning ≥2.4, Transformers ≥4.46, PyTorch Geometric ≥2.7
- **Data:** Polars ≥1.9 (high-performance DataFrames), PyArrow ≥17.0
- **Configuration:** Pydantic ≥2.12 (config models), Hydra-style YAML configs
- **CLI:** Typer ≥0.12 with Rich ≥13.9 formatting
- **Documentation:** MkDocs ≥1.6 with Material theme ≥9.7
- **Development:** pytest ≥8.3, ruff ≥0.6, yapf ≥0.43

## Architecture Summary

The system consists of **four sequential stages**:

1. **Multi-Channel Text Encoding** (`text_model/encoder.py`)
   - Independent LoRA-adapted transformer encoders for title, description, examples, exclusions
   - Base model: sentence-transformers/all-MiniLM-L6-v2
   - Produces 4 Euclidean embeddings per NAICS code

2. **Mixture-of-Experts Fusion** (`text_model/moe.py`)
   - Top-2 gating with load-balancing loss
   - Adaptively fuses the 4 channel embeddings into a single Euclidean embedding

3. **Hyperbolic Contrastive Learning** (`text_model/naics_model.py`, `text_model/loss.py`)
   - Projects embeddings into Lorentz-model hyperbolic space
   - Uses Decoupled Contrastive Learning (DCL) with Lorentzian geodesic distances
   - Includes false negative mitigation via configurable strategies
   - Dynamic Structure-Aware Curriculum (SADC) adapts training difficulty

4. **Hyperbolic Graph Convolutional Refinement** (`graph_model/hgcn.py`)
   - Applies HGCN layers to enforce NAICS parent-child graph structure
   - 4-phase metric-driven curriculum learning system
   - Event-driven architecture with adaptive loss and specialized samplers
   - Combines hyperbolic triplet loss with confidence-based margin adaptation

**Final Output:** High-fidelity Lorentz-model hyperbolic embeddings suitable for hierarchical
search, clustering, and downstream ML tasks.

## Directory Structure

```bash
naics-embedder/
├── src/naics_embedder/       # Main source code (77 Python files)
│   ├── cli/                  # CLI entry point and command groups
│   │   ├── cli.py            # Top-level Typer app
│   │   ├── commands/         # Command implementations
│   │   │   ├── data.py       # Data preparation commands
│   │   │   ├── tools.py      # Utility tools commands
│   │   │   └── training.py   # Training command (661 lines)
│   │   └── __init__.py
│   ├── data/                 # Data preprocessing and generation
│   │   ├── download_data.py  # Download and preprocess NAICS data
│   │   ├── compute_relations.py   # Compute relationship measures
│   │   ├── compute_distances.py   # Compute graph distance measures
│   │   └── create_triplets.py     # Create contrastive training triplets
│   ├── text_model/           # Stage 1-3: Text encoding and contrastive learning
│   │   ├── encoder.py        # Multi-channel LoRA encoder
│   │   ├── moe.py            # Mixture-of-Experts implementation
│   │   ├── hyperbolic.py     # Lorentz ops, exponential/log maps
│   │   ├── naics_model.py    # PyTorch Lightning module (2,030 lines) ⭐
│   │   ├── loss.py           # DCL loss, hierarchy loss, rank-order loss
│   │   ├── curriculum.py     # Structure-Aware Dynamic Curriculum
│   │   ├── hard_negative_mining.py    # Hard negative mining strategies
│   │   ├── hyperbolic_clustering.py   # False negative clustering
│   │   ├── false_negative_strategies.py  # FN strategy configuration
│   │   ├── evaluation.py     # Embedding evaluation metrics
│   │   └── dataloader/       # Data loading subsystem
│   │       ├── streaming_dataset.py   # Streaming Polars datasets
│   │       ├── datamodule.py          # PyTorch Lightning DataModule
│   │       └── tokenization_cache.py  # Disk-based tokenization cache
│   ├── graph_model/          # Stage 4: HGCN refinement
│   │   ├── hgcn.py           # Hyperbolic graph convolutional network
│   │   ├── evaluation.py     # HGCN evaluation metrics
│   │   ├── curriculum/       # ⭐ Advanced 4-phase curriculum system
│   │   │   ├── controller.py        # CurriculumController, phase transitions
│   │   │   ├── event_bus.py         # Event-driven architecture
│   │   │   ├── adaptive_loss.py     # MACL adaptive loss computation
│   │   │   ├── sampling.py          # Phase-specific sampling strategies
│   │   │   ├── monitoring.py        # Training progress analysis & reports
│   │   │   └── preprocess_curriculum.py  # Curriculum data preprocessing
│   │   └── dataloader/       # Graph data loading
│   │       ├── hgcn_streaming_dataset.py  # Graph streaming dataset
│   │       └── hgcn_datamodule.py         # Graph data module
│   ├── tools/                # Utility tools for config, metrics, visualization
│   │   ├── config_tools.py
│   │   ├── metrics_tools.py
│   │   ├── _visualize_metrics.py      # Training visualization (executable)
│   │   └── _investigate_hierarchy.py  # Hierarchy investigation
│   └── utils/                # Backend utilities, config, console
│       ├── backend.py        # Device selection, GPU memory detection
│       ├── config.py         # Pydantic config models (1,042 lines) ⭐
│       ├── console.py        # Rich console logging, table formatting
│       ├── hyperbolic.py     # LorentzManifold, CurvatureManager
│       ├── training.py       # Hardware detection, checkpoint resolution
│       ├── validation.py     # Data & config validation system
│       ├── warnings.py       # Centralized warning management
│       └── utilities.py      # General helper functions
├── tests/                    # Comprehensive test suite (~9,093 lines)
│   ├── conftest.py           # Pytest fixtures
│   ├── unit/                 # 19 unit test files
│   │   ├── test_config.py
│   │   ├── test_curriculum.py
│   │   ├── test_datamodule.py
│   │   ├── test_encoder.py
│   │   ├── test_evaluation.py
│   │   ├── test_graph_curriculum.py       # Graph curriculum tests
│   │   ├── test_graph_preprocessing.py
│   │   ├── test_hard_negative_mining.py
│   │   ├── test_hyperbolic.py
│   │   ├── test_loss.py
│   │   ├── test_moe.py
│   │   ├── test_naics_model.py
│   │   ├── test_streaming_dataset.py
│   │   ├── test_tokenization_cache.py
│   │   └── ...
│   └── integration/          # Integration tests (placeholder)
├── conf/                     # Configuration files
│   ├── config.yaml           # Base training configuration
│   ├── data/                 # Data generation configs
│   │   ├── download.yaml
│   │   ├── relations.yaml
│   │   ├── distances.yaml
│   │   └── triplets.yaml
│   └── data_loader/          # Data loading configs
│       └── tokenization.yaml
├── docs/                     # MkDocs documentation
│   ├── .nav.yml              # Navigation configuration
│   ├── index.md
│   ├── overview.md
│   ├── quickstart.md
│   ├── usage.md
│   ├── text_training.md
│   ├── hgcn_training.md
│   ├── benchmarks.md
│   └── api/                  # 27 API reference files (auto-generated)
├── scripts/                  # Utility scripts
│   ├── format_code.sh        # Run ruff and yapf formatting
│   └── create_refactor_issues.sh
├── outputs/                  # Training outputs and visualizations
│   ├── visualizations/       # Comparative training visualizations
│   ├── 01_text/, 02_text/, 03_text/, ...  # Experiment directories
│   └── ...
├── reports/                  # Generated analysis reports
│   ├── SADC.md               # Structure-Aware Dynamic Curriculum analysis
│   ├── hgcn.md               # HGCN analysis
│   ├── hgcn_curriculum.md    # HGCN curriculum analysis
│   ├── distance_stats.pdf
│   ├── relation_stats.pdf
│   └── triplets_stats.pdf
├── data/                     # Generated data files (gitignored)
│   ├── naics_descriptions.parquet
│   ├── naics_relations.parquet
│   ├── naics_distances.parquet
│   └── naics_training_pairs.parquet
├── checkpoints/              # Model checkpoints (gitignored)
├── logs/                     # Training logs (gitignored)
├── .github/workflows/        # CI/CD workflows
│   ├── docs.yml              # Build and deploy docs to GitHub Pages
│   └── tests.yml             # Run pytest with coverage, ruff linting
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
├── mkdocs.yml                # Documentation config
├── .gitignore
├── .markdownlint.jsonc       # Markdown linting rules
├── scratch.ipynb             # Experimentation notebook
└── README.md
```

## Key Concepts and Patterns

### 1. Hyperbolic Geometry (Lorentz Model)

The system implements **two levels** of hyperbolic geometry support:

**Core Operations (`text_model/hyperbolic.py`):**

- `LorentzOps` class - Low-level Lorentz model operations
- **Lorentz Inner Product:** `<x, y>_L = -x[0] * y[0] + x[1:] · y[1:]`
- **Geodesic Distance:** `d(x, y) = arcosh(-<x, y>_L)`
- **Exponential Map:** Projects tangent vectors from tangent space to hyperboloid
- **Logarithmic Map:** Projects hyperboloid points to tangent space

**Manifold Abstraction (`utils/hyperbolic.py`):**

- `LorentzManifold` class - High-level manifold interface
- `CurvatureManager` - Learnable or fixed curvature management
- `ManifoldAdapter` - Compatibility layer for different hyperbolic implementations
- `validate_hyperbolic_embeddings` - Numerical validation utilities

### 2. Multi-Channel Architecture

Each NAICS code has **4 text channels**:

- **Title:** Short name (e.g., "Computer Systems Design Services")
- **Description:** Detailed description of the industry
- **Examples:** Example activities and products
- **Excluded:** Related but excluded activities

Each channel is encoded by a **separate LoRA-adapted transformer** (via PEFT library) to capture
channel-specific semantics.

### 3. Mixture-of-Experts (MoE)

- **Top-k Gating:** Routes each input to the top-k most relevant experts (k=2)
- **Load Balancing:** Auxiliary loss ensures even expert utilization
- **Implementation:** `text_model/moe.py`
- **Purpose:** Learns adaptive fusion of the 4 channel embeddings

### 4. Dynamic Structure-Aware Curriculum Learning (SADC)

**Text Model Curriculum** uses a dynamic, structure-aware approach that adapts based on the
hierarchical structure of NAICS codes:

- Dynamically adjusts training difficulty based on code relationships
- Leverages tree distance and exclusion relationships
- Adapts negative sampling strategy based on training progress
- Three-phase system with dynamic transitions

**Key Files:**

- `text_model/curriculum.py` - Dynamic curriculum scheduler implementation
- `text_model/dataloader/streaming_dataset.py` - Structure-aware sampling

### 5. False Negative Mitigation

**Problem:** In contrastive learning, some "negative" samples may actually be semantically
similar to the anchor (false negatives), harming training.

**Solutions:** The system provides **configurable false negative strategies**:

1. **Eliminate:** Mask false negatives from contrastive denominator (set similarity to `-inf`)
2. **Attract:** Apply auxiliary loss to attract false negatives to anchor
3. **Hybrid:** Combine both strategies

**Detection:** Curriculum-based clustering periodically clusters embeddings to identify false
negatives sharing the same cluster as anchor.

**Key Files:**

- `text_model/false_negative_strategies.py` - Strategy configuration
- `text_model/hyperbolic_clustering.py` - Clustering-based detection

### 6. Decoupled Contrastive Learning (DCL)

Standard InfoNCE loss can have gradient issues. DCL decouples positive and negative terms:

```python
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

This provides better gradient flow and numerical stability.

**Implementation:** `text_model/loss.py` - `HyperbolicInfoNCELoss`

### 7. Advanced Graph Curriculum System

The **HGCN refinement stage** uses a sophisticated **4-phase metric-driven curriculum**:

**Architecture:** Event-driven with centralized controller

**Phases:**

1. **Anchoring** - Hub nodes, uniform sampling, high curvature
2. **Expansion** - Tail nodes, adaptive margins (MACL)
3. **Discrimination** - Hard negative mining
4. **Stabilization** - Full graph, knowledge distillation

**Key Components:**

- **Controller** (`curriculum/controller.py`) - Phase transitions based on validation metrics
- **Event Bus** (`curriculum/event_bus.py`) - Event-driven coordination
- **Adaptive Loss** (`curriculum/adaptive_loss.py`) - MACL (Margin-Adaptive Curriculum Learning)
  with confidence tracking
- **Sampling** (`curriculum/sampling.py`) - Phase-specific samplers (Hub, Difficulty-Weighted,
  Hard Negative, Blended)
- **Monitoring** (`curriculum/monitoring.py`) - Progress analysis, HTML/Markdown reports,
  visualizations
- **Preprocessing** (`curriculum/preprocess_curriculum.py`) - Node scoring, difficulty thresholds,
  relation cardinality

### 8. Distributed Training (Multi-GPU)

The codebase supports **global batch sampling** across multiple GPUs:

- Uses `torch.distributed.all_gather` to collect embeddings from all ranks
- Enables hard negative mining across the global batch
- **Critical:** Gradients flow back through `all_gather` operation

**Implementation:** `text_model/naics_model.py` - `gather_embeddings_global()`

### 9. Validation and Configuration System

**Validation System** (`utils/validation.py`):

- Data path validation
- Parquet schema validation (descriptions, distances, relations, triplets)
- Training configuration validation
- Tokenization cache validation
- Returns `ValidationResult` with warnings and errors
- `require_valid_config` - Raises on validation failure

**Configuration System** (`utils/config.py`):

- Pydantic-based configuration models with validation
- Comprehensive config classes for all subsystems
- `load_config` function with environment variable support
- Hydra-style YAML configuration files

### 10. Hardware Detection and Optimization

**Training Utilities** (`utils/training.py`):

- `detect_hardware` - Automatic GPU/CPU/MPS detection
- `get_gpu_memory_info` - Memory statistics
- `HardwareInfo` dataclass - Capability tracking
- `CheckpointInfo` - Checkpoint metadata
- `TrainingResult` - Training outcome tracking
- `create_trainer` - PyTorch Lightning Trainer factory
- `resolve_checkpoint` - Smart checkpoint path resolution

**Backend** (`utils/backend.py`):

- Device selection (CUDA, MPS, CPU)
- GPU memory detection
- Directory setup utilities

## Development Setup

### Initial Setup

```bash
# Clone repository
git clone https://github.com/lowmason/naics-embedder.git
cd naics-embedder

# Install uv (if not already installed)
pip install uv

# Install dependencies
uv sync
```

### Running Commands

All commands use the `naics-embedder` CLI via `uv run`:

```bash
# Show help
uv run naics-embedder --help

# Data preparation commands
uv run naics-embedder data preprocess  # Download and preprocess NAICS data
uv run naics-embedder data relations   # Compute relation metrics
uv run naics-embedder data distances   # Compute distance metrics
uv run naics-embedder data triplets    # Create training triplets
uv run naics-embedder data all         # Run all data preparation steps

# Training commands
uv run naics-embedder train            # Train model
uv run naics-embedder train --config conf/my_config.yaml  # Custom config

# Tools commands
uv run naics-embedder tools config     # Show current configuration
uv run naics-embedder tools visualize  # Visualize training metrics
uv run naics-embedder tools investigate  # Investigate hierarchy correlation
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=naics_embedder

# Run specific test file
uv run pytest tests/unit/test_encoder.py

# Run with parallelization
uv run pytest -n auto
```

## Code Style and Conventions

### Python Formatting

**Tools:**

- **Ruff:** Linting and import sorting
- **YAPF:** Detailed formatting (spacing, chaining, alignment)

**Key Rules (from `pyproject.toml`):**

- **Line length:** 100 characters
- **Quotes:** Single quotes preferred (`'` not `"`)
- **Blank lines:** 1 after top-level definitions
- **Indentation:** 4 spaces
- **Method chaining:** Dot-aligned, split before dot
- **Imports:** Sorted with `ruff` (E, F, I, Q rules)

**Example:**

```python
# Good: single quotes, dot-aligned chaining
result = (
    df
    .filter(pl.col('code').is_not_null())
    .select(['code', 'title', 'description'])
    .collect()
)

# Bad: double quotes, no alignment
result = df.filter(pl.col("code").is_not_null()).select(["code", "title", "description"]).collect()
```

**Format Code:**

```bash
# Use the formatting script (recommended)
./scripts/format_code.sh

# Or manually:
uv run yapf -i -r src/
uv run ruff check src/
uv run ruff format src/
```

### Markdown Formatting

**Rules (from `.markdownlint.jsonc`):**

- **Line length:** 100 characters (code blocks and tables exempt)
- **Headings:** 1 blank line above and below
- **Lists:** Indent by 2 spaces
- **Max consecutive blank lines:** 1

### Section Dividers

Python files use **semantic section dividers** with consistent formatting:

```python
# -------------------------------------------------------------------------------------------------
# Section Title
# -------------------------------------------------------------------------------------------------

# Code here...
```

**Common sections:**

- Imports and settings
- Config / Constants
- Utilities
- Main logic
- Entry point

### Logging

Use Python's `logging` module (not `print`):

```python
import logging

logger = logging.getLogger(__name__)

logger.info('Starting training...')
logger.warning('Curvature out of bounds, clamping to safe range')
logger.error('Failed to load checkpoint')
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Dict, List, Tuple

def compute_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float = 1.0
) -> torch.Tensor:
    ...
```

### Docstrings

Use **single-quote triple-quoted docstrings**:

```python
def exp_map_zero(x_tan: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    '''
    Exponential map from tangent space at origin to Lorentz hyperboloid.

    Args:
        x_tan: Tangent vector (N, D)
        c: Curvature (positive scalar)

    Returns:
        Hyperbolic point on hyperboloid (N, D+1)
    '''
    ...
```

## Common Development Tasks

### 1. Adjusting Training Configuration

```bash
# Edit the main configuration file
# Adjust hyperparameters (learning_rate, max_epochs, batch_size, etc.)
# in conf/config.yaml

# Run training with modified config
uv run naics-embedder train

# Override specific parameters at runtime (Hydra-style)
uv run naics-embedder train training.learning_rate=1e-4 data_loader.batch_size=16
```

**Note:** The project uses dynamic Structure-Aware Curriculum Learning. Training difficulty
adapts automatically based on the NAICS hierarchical structure and training progress.

### 2. Modifying Loss Functions

**Location:** `src/naics_embedder/text_model/loss.py`

**Key losses:**

- `HyperbolicInfoNCELoss` - Decoupled contrastive loss in Lorentz space
- `HierarchyPreservationLoss` - Encourages distance correlation with ground truth
- `RankOrderPreservationLoss` - Spearman rank correlation loss

**Steps:**

1. Edit the loss class in `loss.py`
2. Adjust loss weights in `conf/config.yaml` under `loss:` section
3. Re-run training to test changes

### 3. Adding New Evaluation Metrics

**Location:** `src/naics_embedder/text_model/evaluation.py`

**Existing metrics:**

- `EmbeddingStatistics` - Radius, norm, diversity
- `HierarchyMetrics` - Distance correlation, MAP, level consistency

**Steps:**

1. Add new metric computation in `evaluation.py`
2. Update `EmbeddingEvaluator.evaluate()` to call new metric
3. Modify `naics_model.py` validation step to log new metric
4. Add corresponding test in `tests/unit/test_evaluation.py`

### 4. Debugging Training Issues

**Check hyperbolic validity:**

```python
from naics_embedder.utils.hyperbolic import validate_hyperbolic_embeddings

# After computing embeddings
is_valid = validate_hyperbolic_embeddings(embeddings_hyp, curvature=1.0, tolerance=1e-5)
if not is_valid:
    logger.warning('Embeddings not on Lorentz manifold!')
```

**Investigate low hierarchy correlation:**

```bash
uv run naics-embedder tools investigate
```

**Visualize training metrics:**

```bash
uv run naics-embedder tools visualize
```

**Check validation results:**

```python
from naics_embedder.utils.validation import validate_data_paths, validate_training_config

# Validate data files
result = validate_data_paths('data')
if not result.is_valid:
    print(result.errors)

# Validate training config
result = validate_training_config(config)
if not result.is_valid:
    print(result.errors)
```

### 5. Working with Configuration

**Pydantic Configuration System:**

The project uses Pydantic for type-safe configuration with automatic validation.

**Load and validate config:**

```python
from naics_embedder.utils.config import load_config, Config

config = load_config('conf/config.yaml')  # Automatically validated
```

**Override config at runtime:**

```bash
# Override single value
uv run naics-embedder train training.learning_rate=2e-4

# Override multiple values
uv run naics-embedder train \
  training.learning_rate=2e-4 \
  data_loader.batch_size=16 \
  loss.hierarchy_weight=0.5
```

**View effective config:**

```bash
uv run naics-embedder tools config
```

### 6. Working with Graph Curriculum System

**Preprocess curriculum data:**

```python
from naics_embedder.graph_model.curriculum import preprocess_curriculum_data

# Generate node scores and difficulty thresholds
preprocess_curriculum_data(
    distances_path='data/naics_distances.parquet',
    relations_path='data/naics_relations.parquet',
    output_dir='data'
)
```

**Monitor curriculum training:**

```python
from naics_embedder.graph_model.curriculum import CurriculumAnalyzer

analyzer = CurriculumAnalyzer()
# .. during training ..
report = analyzer.generate_report(output_path='reports/curriculum.md')
```

**Customize curriculum phases:**

Edit phase configurations in `conf/config.yaml` under `graph_curriculum:` section or use the
controller programmatically.

## Important Implementation Details

### 1. Gradient Checkpointing

**Enabled by default** to save GPU memory:

```python
# In encoder.py
if use_gradient_checkpointing:
    for channel in self.channels:
        self.encoders[channel].enable_input_require_grads()
        self.encoders[channel].base_model.gradient_checkpointing_enable()
```

**Trade-off:** Reduces memory usage at the cost of ~20% slower training.

### 2. Curvature Management

**Two approaches:**

1. **Clamped curvature** (`text_model/hyperbolic.py`) - Clamps to safe range `[0.1, 10.0]`
2. **Managed curvature** (`utils/hyperbolic.py`) - `CurvatureManager` with learnable or fixed
   curvature

Both prevent numerical instability in hyperbolic operations.

### 3. Mixed Precision Training

**Enabled by default** via PyTorch Lightning:

```yaml
# In conf/config.yaml
training:
  trainer:
    precision: "16-mixed"
```

**Benefits:** ~2x speedup and ~50% memory reduction on modern GPUs.

### 4. Streaming Datasets

Data is loaded via **streaming Polars datasets** to handle large-scale data efficiently:

- **Implementation:** `text_model/dataloader/streaming_dataset.py`
- **Benefits:** Low memory footprint, fast random access via PyArrow
- **Format:** Parquet files with efficient columnar storage

### 5. Tokenization Caching

Tokenized inputs are **cached to disk** to avoid re-tokenization:

- **Implementation:** `text_model/dataloader/tokenization_cache.py`
- **Cache location:** `data/*.cache`
- **Invalidation:** Automatic on tokenizer or data changes (hash-based)

### 6. Checkpoint Management

**Checkpoint structure:**

```bash
checkpoints/
├── <experiment_name>/
│   ├── last.ckpt                       # Last checkpoint (for resuming)
│   ├── naics-epoch=X-val_loss=Y.ckpt  # Checkpoints by epoch/loss
│   └── config.yaml                     # Config snapshot for this run
```

**Resume training:**

```bash
# Resume from last checkpoint
uv run naics-embedder train --ckpt-path last

# Resume from specific checkpoint
uv run naics-embedder train --ckpt-path checkpoints/my_experiment/naics-epoch=5-val_loss=0.1234.ckpt
```

### 7. Warning Management

The system uses **centralized warning suppression** (`utils/warnings.py`):

- Suppresses known benign warnings from dependencies
- Documents rationale for each suppressed warning
- Applied globally via `cli/cli.py` at startup

```python
from naics_embedder.utils.warnings import configure_warnings, list_suppressed_warnings

configure_warnings()  # Apply all suppressions
warnings_list = list_suppressed_warnings()  # Get list of suppressed warnings
```

## Testing and Validation

### Test Suite

The project has a comprehensive test suite with **~9,093 lines** of test code across **19 unit
test files**:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=naics_embedder --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_encoder.py

# Run tests in parallel
uv run pytest -n auto

# Run tests with verbose output
uv run pytest -v
```

### Key Test Files

- `test_config.py` - Configuration validation
- `test_curriculum.py` - Text curriculum
- `test_graph_curriculum.py` - Graph curriculum system (all components)
- `test_datamodule.py` - Data loading
- `test_encoder.py` - Multi-channel encoding
- `test_evaluation.py` - Evaluation metrics
- `test_hyperbolic.py` - Lorentz operations
- `test_loss.py` - Loss functions
- `test_moe.py` - Mixture-of-Experts
- `test_naics_model.py` - Main model
- `test_streaming_dataset.py` - Streaming datasets
- `test_tokenization_cache.py` - Tokenization cache

### Manual Testing

```bash
# Test data pipeline
uv run naics-embedder data all

# Test training (quick)
uv run naics-embedder train training.trainer.max_epochs=2

# Validate configuration
uv run naics-embedder tools config
```

### Validation Metrics

During training, the model computes validation metrics every epoch:

- **Hyperbolic radius:** Mean radius of embeddings (should be stable)
- **Hierarchy correlation:** Spearman correlation between learned and ground-truth distances
- **Mean Average Precision (MAP):** Retrieval quality
- **Diversity:** Average pairwise distance (should not collapse)

**Expected values (after convergence):**

- Hierarchy correlation: > 0.7
- MAP: > 0.6
- Mean radius: 2.0-4.0 (depends on data and curvature)

## CI/CD and Documentation

### GitHub Actions

**Workflows:**

1. **Documentation** (`.github/workflows/docs.yml`)
   - **Trigger:** Push to `main` or `master` branch
   - **Action:** Build and deploy MkDocs documentation to GitHub Pages
   - **Output:** <https://lowmason.github.io/naics-embedder/>

2. **Tests** (`.github/workflows/tests.yml`)
   - **Trigger:** Push, pull request
   - **Action:** Run pytest with coverage (Python 3.10, 3.12), ruff linting
   - **Reports:** Coverage reports uploaded to artifacts

### Documentation

**Build locally:**

```bash
# Build docs
uv run mkdocs build

# Serve locally with live reload
uv run mkdocs serve
```

**View at:** <http://localhost:8000>

**Update API docs:**

API documentation is **auto-generated** from docstrings using `mkdocstrings`. The system has
**27 API reference pages**.

To add new module to docs:

1. Add markdown file in `docs/api/`
2. Include mkdocstrings directive:

```markdown
# Module Name

::: naics_embedder.module_name
```

## Common Pitfalls and Solutions

### 1. Embeddings Not on Lorentz Manifold

**Symptom:** Warning during training: `"Embeddings not on Lorentz manifold"`

**Causes:**

- Numerical instability in exponential map
- Gradient explosion
- Learning rate too high
- Curvature out of safe range

**Solutions:**

```bash
# Reduce learning rate
uv run naics-embedder train training.learning_rate=1e-5

# Increase gradient clipping
uv run naics-embedder train training.trainer.gradient_clip_val=0.5

# Enable mixed precision (if not already)
uv run naics-embedder train training.trainer.precision="16-mixed"
```

**Debug:**

```python
from naics_embedder.utils.hyperbolic import validate_hyperbolic_embeddings

is_valid = validate_hyperbolic_embeddings(embeddings, curvature=1.0, tolerance=1e-5)
```

### 2. Low Hierarchy Correlation

**Symptom:** `hierarchy_corr` metric remains low (<0.3)

**Causes:**

- Insufficient training
- Loss weights not balanced
- Ground truth distances not informative
- Curriculum not adapting properly

**Solutions:**

```bash
# Investigate ground truth distances
uv run naics-embedder tools investigate

# Increase hierarchy loss weight
uv run naics-embedder train loss.hierarchy_weight=0.5

# Train longer
uv run naics-embedder train training.trainer.max_epochs=30

# Adjust curriculum settings
uv run naics-embedder train curriculum.enabled=true
```

### 3. OOM (Out of Memory) Errors

**Solutions:**

```bash
# Reduce batch size
uv run naics-embedder train data_loader.batch_size=8

# Increase gradient accumulation
uv run naics-embedder train training.trainer.accumulate_grad_batches=4

# Enable gradient checkpointing (if not already)
uv run naics-embedder train model.use_gradient_checkpointing=true

# Use mixed precision (if not already)
uv run naics-embedder train training.trainer.precision="16-mixed"
```

**Hardware detection:**

```python
from naics_embedder.utils.training import detect_hardware, get_gpu_memory_info

hw_info = detect_hardware()
gpu_info = get_gpu_memory_info()
print(f'Available GPU memory: {gpu_info["available_gb"]:.2f} GB')
```

### 4. Checkpoint Not Found

**Symptom:** `FileNotFoundError: checkpoint not found`

**Solutions:**

```bash
# Check checkpoint directory exists
ls checkpoints/<experiment_name>/

# Use checkpoint resolution
uv run naics-embedder train --ckpt-path last  # Auto-resolves to latest

# Use absolute path
uv run naics-embedder train --ckpt-path /absolute/path/to/checkpoint.ckpt
```

**Programmatic resolution:**

```python
from naics_embedder.utils.training import resolve_checkpoint

ckpt_path = resolve_checkpoint('last', 'checkpoints/my_experiment')
```

### 5. Validation Errors

**Symptom:** Errors during data loading or config parsing

**Solutions:**

```python
from naics_embedder.utils.validation import (
    validate_data_paths,
    validate_training_config,
    require_valid_config
)

# Check data files
result = validate_data_paths('data')
if not result.is_valid:
    print('Errors:', result.errors)
    print('Warnings:', result.warnings)

# Validate config
result = validate_training_config(config)
require_valid_config(result)  # Raises if invalid
```

## Git Workflow

### Branch Naming

Follow the pattern: `claude/<descriptive-name>-<session-id>`

**Examples:**

- `claude/update-claude-md-01FgsKX3pMhy1GMWM6ivoh4U`
- `claude/add-graph-curriculum-ABC123XYZ`

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add hierarchy preservation loss to training loop"
git commit -m "Fix gradient flow issue in hyperbolic convolution"
git commit -m "Implement 4-phase graph curriculum system"
git commit -m "Update CLAUDE.md to reflect current codebase structure"

# Bad
git commit -m "fix bug"
git commit -m "updates"
git commit -m "yo brah!"
```

### Push Workflow

Always push to the designated Claude branch:

```bash
# Push with upstream tracking
git push -u origin claude/update-claude-md-01FgsKX3pMhy1GMWM6ivoh4U

# If push fails due to network, retry with exponential backoff
# (2s, 4s, 8s, 16s - up to 4 retries)
```

## File Modification Guidelines

### When to Edit vs. Create

**ALWAYS prefer editing** existing files over creating new ones:

- Modifying loss functions → Edit `text_model/loss.py`
- Adding CLI command → Edit `cli/commands/*.py`
- Adjusting config → Edit `conf/config.yaml`
- Adding utilities → Edit appropriate `utils/*.py`

**ONLY create new files** when:

- Adding entirely new module (e.g., new model architecture)
- Adding new data preprocessing script
- Creating new test file
- Adding new documentation page

### Key Files to Edit

**Configuration:**

- `conf/config.yaml` - Base training configuration
- `conf/data/*.yaml` - Data generation configs
- `conf/data_loader/tokenization.yaml` - Tokenization config

**Model Architecture:**

- `text_model/encoder.py` - Multi-channel encoder
- `text_model/moe.py` - Mixture-of-Experts
- `text_model/loss.py` - Loss functions
- `graph_model/hgcn.py` - Hyperbolic GCN
- `graph_model/curriculum/*.py` - Graph curriculum system

**Training:**

- `text_model/naics_model.py` - Main PyTorch Lightning module (2,030 lines)
- `cli/commands/training.py` - Training CLI commands (661 lines)

**Data:**

- `data/*.py` - Data preprocessing scripts

**Utils:**

- `utils/config.py` - Configuration models (1,042 lines)
- `utils/training.py` - Training utilities
- `utils/validation.py` - Validation system
- `utils/hyperbolic.py` - Hyperbolic manifold utilities

## Summary Checklist for AI Assistants

When working on this codebase:

- [ ] Use `uv run` for all CLI commands
- [ ] Follow single-quote Python style (`'` not `"`)
- [ ] Keep line length ≤ 100 characters
- [ ] Use semantic section dividers in Python files
- [ ] Add type hints to function signatures
- [ ] Use `logging` instead of `print`
- [ ] Write unit tests for new functionality in `tests/unit/`
- [ ] Run tests before committing: `uv run pytest`
- [ ] Format code: `./scripts/format_code.sh` or manually with yapf/ruff
- [ ] Test changes with a quick training run: `uv run naics-embedder train training.trainer.max_epochs=2`
- [ ] Check hyperbolic validity when modifying geometry code
- [ ] Use validation utilities to check data and config
- [ ] Update configuration files (not hardcoded values) for hyperparameters
- [ ] Commit with descriptive messages (not "fix bug" or "yo brah!")
- [ ] Push to the designated Claude branch with `-u origin <branch-name>`
- [ ] Update documentation if adding new features or changing APIs

## Architecture Decision Records

### Why Two Hyperbolic Implementations?

- **`text_model/hyperbolic.py`** - Low-level operations used during text model training
- **`utils/hyperbolic.py`** - High-level abstraction for general use, validation, curvature
  management

This separation allows the text model to use optimized operations while providing a clean
interface for other components.

### Why Pydantic + YAML (not pure Hydra)?

- **Pydantic** provides strong type validation and IDE support
- **YAML files** provide human-readable configuration
- **Best of both worlds:** Type safety + flexibility

### Why Two Curriculum Systems?

- **Text Model (SADC):** Optimized for contrastive learning on text embeddings
- **Graph Model (4-phase):** Optimized for graph convolutions with explicit phase transitions

Different training dynamics require different curriculum strategies.

## Additional Resources

- **README.md:** High-level architecture overview
- **docs/quickstart.md:** Quick start guide
- **docs/usage.md:** Complete CLI command reference
- **docs/text_training.md:** Detailed text model training guide
- **docs/hgcn_training.md:** HGCN refinement guide
- **docs/benchmarks.md:** Performance benchmarks
- **API Docs:** <https://lowmason.github.io/naics-embedder/> (27 auto-generated pages)
- **Reports:** See `reports/` for generated analysis (SADC, HGCN curriculum, etc.)

## Contact and Support

- **Repository:** <https://github.com/lowmason/naics-embedder>
- **Author:** Lowell Mason
- **Issues:** Report bugs or request features via GitHub Issues
- **Documentation:** <https://lowmason.github.io/naics-embedder/>
