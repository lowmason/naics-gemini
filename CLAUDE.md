# CLAUDE.md - AI Assistant Guide for NAICS Embedder

## Project Overview

**NAICS Hyperbolic Embedding System** is a awesome sophisticated machine learning framework that produces
hyperbolic embeddings for the North American Industry Classification System (NAICS). The system
combines multi-channel text encoding, Mixture-of-Experts fusion, hyperbolic contrastive learning,
and hyperbolic graph refinement to create geometry-aware embeddings aligned with the hierarchical
NAICS taxonomy.

**Key Technologies:**

- **Language:** Python 3.10+
- **Package Manager:** `uv` (modern Python package manager)
- **ML Frameworks:** PyTorch, PyTorch Lightning, Transformers (HuggingFace), PyTorch Geometric
- **Data:** Polars (high-performance DataFrames), PyArrow
- **Configuration:** Hydra (hierarchical configuration management)
- **CLI:** Typer with Rich formatting
- **Documentation:** MkDocs with Material theme

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
   - Includes false negative mitigation via curriculum-based clustering

4. **Hyperbolic Graph Convolutional Refinement** (`graph_model/hgcn.py`)
   - Applies HGCN layers to enforce NAICS parent-child graph structure
   - Combines hyperbolic triplet loss with per-level radial regularization

**Final Output:** High-fidelity Lorentz-model hyperbolic embeddings suitable for hierarchical
search, clustering, and downstream ML tasks.

## Directory Structure

```bash
naics-embedder/
├── src/naics_embedder/       # Main source code
│   ├── cli/                  # CLI entry point and command groups
│   │   ├── commands/         # data.py, tools.py, training.py
│   │   └── __init__.py
│   ├── data/                 # Data preprocessing and generation
│   │   ├── download_data.py
│   │   ├── compute_relations.py
│   │   ├── compute_distances.py
│   │   └── create_triplets.py
│   ├── text_model/           # Stage 1-3: Text encoding and contrastive learning
│   │   ├── encoder.py        # Multi-channel encoder + MoE + hyperbolic projection
│   │   ├── naics_model.py    # PyTorch Lightning module for training
│   │   ├── loss.py           # DCL loss, hierarchy loss, rank-order loss
│   │   ├── moe.py            # Mixture-of-Experts implementation
│   │   ├── hyperbolic.py     # Lorentz ops, exponential/log maps
│   │   ├── curriculum.py     # Curriculum scheduler
│   │   ├── hard_negative_mining.py
│   │   ├── hyperbolic_clustering.py
│   │   ├── evaluation.py     # Embedding evaluation metrics
│   │   └── dataloader/       # Streaming dataset, tokenization cache
│   ├── graph_model/          # Stage 4: HGCN refinement
│   │   ├── hgcn.py           # Hyperbolic graph convolutional network
│   │   ├── evaluation.py
│   │   └── dataloader/       # Graph data loading
│   ├── tools/                # Utility tools for config, metrics, visualization
│   │   ├── config_tools.py
│   │   ├── metrics_tools.py
│   │   ├── _visualize_metrics.py
│   │   └── _investigate_hierarchy.py
│   └── utils/                # Backend utilities, config, console
│       ├── backend.py        # Device selection, directory setup
│       ├── config.py         # Pydantic config models
│       ├── console.py        # Rich console utilities
│       └── utilities.py      # General utilities
├── conf/                     # Hydra configuration files
│   ├── config.yaml           # Base configuration
│   ├── data/                 # Data generation configs
│   └── data_loader/          # Tokenization configs
├── docs/                     # MkDocs documentation
│   ├── index.md
│   ├── overview.md
│   ├── usage.md
│   ├── text_training.md
│   ├── hgcn_training.md
│   ├── literature.md
│   └── api/                  # Auto-generated API docs
├── data/                     # Generated data files (gitignored)
│   ├── naics_descriptions.parquet
│   ├── naics_relations.parquet
│   ├── naics_distances.parquet
│   └── naics_training_pairs.parquet
├── checkpoints/              # Model checkpoints (gitignored)
├── outputs/                  # Training outputs and visualizations
├── logs/                     # Training logs (gitignored)
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
├── mkdocs.yml                # Documentation config
├── .gitignore
├── .markdownlint.jsonc       # Markdown linting rules
└── README.md
```

## Key Concepts and Patterns

### 1. Hyperbolic Geometry (Lorentz Model)

- **Lorentz Model:** Hyperboloid model of hyperbolic space, represented by points on the upper
  sheet of a two-sheeted hyperboloid
- **Lorentz Inner Product:** `<x, y>_L = -x[0] * y[0] + x[1:] · y[1:]`
- **Geodesic Distance:** `d(x, y) = arcosh(-<x, y>_L)`
- **Exponential Map:** Projects tangent vectors from tangent space to hyperboloid
- **Logarithmic Map:** Projects hyperboloid points to tangent space
- **Implementation:** `text_model/hyperbolic.py` contains `LorentzOps` class with all operations

### 2. Multi-Channel Architecture

Each NAICS code has **4 text channels**:

- **Title:** Short name (e.g., "Computer Systems Design Services")
- **Description:** Detailed description of the industry
- **Examples:** Example activities and products
- **Excluded:** Related but excluded activities

Each channel is encoded by a **separate LoRA-adapted transformer** to capture channel-specific
semantics.

### 3. Mixture-of-Experts (MoE)

- **Top-k Gating:** Routes each input to the top-k most relevant experts (k=2)
- **Load Balancing:** Auxiliary loss ensures even expert utilization
- **Implementation:** `text_model/moe.py`
- **Purpose:** Learns adaptive fusion of the 4 channel embeddings

### 4. Dynamic Structure-Aware Curriculum Learning

Training uses a **dynamic, structure-aware curriculum** that adapts based on the hierarchical
structure of NAICS codes:

- Dynamically adjusts training difficulty based on code relationships
- Leverages tree distance and exclusion relationships
- Adapts negative sampling strategy based on training progress

**Key Files:**

- `text_model/curriculum.py` - Dynamic curriculum scheduler implementation
- `data_loader/streaming.py` - Structure-aware sampling configuration

### 5. False Negative Mitigation

**Problem:** In contrastive learning, some "negative" samples may actually be semantically
similar to the anchor (false negatives), harming training.

**Solution:** Curriculum-based clustering to identify and mask false negatives:

1. Periodically cluster embeddings (e.g., KMeans)
2. Identify negatives sharing the same cluster as anchor
3. Mask these from the contrastive denominator (set similarity to `-inf`)

**Implementation:** `text_model/hyperbolic_clustering.py`

### 6. Decoupled Contrastive Learning (DCL)

Standard InfoNCE loss can have gradient issues. DCL decouples positive and negative terms:

```python
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

This provides better gradient flow and numerical stability.

**Implementation:** `text_model/loss.py` - `HyperbolicInfoNCELoss`

### 7. Distributed Training (Multi-GPU)

The codebase supports **global batch sampling** across multiple GPUs:

- Uses `torch.distributed.all_gather` to collect embeddings from all ranks
- Enables hard negative mining across the global batch
- **Critical:** Gradients flow back through `all_gather` operation

**Implementation:** `text_model/naics_model.py` - `gather_embeddings_global()`

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

# Generate data
uv run naics-embedder data all

# Train model
uv run naics-embedder train

# Train with custom config
uv run naics-embedder train --config conf/my_config.yaml

# Show config
uv run naics-embedder tools config

# Visualize metrics
uv run naics-embedder tools visualize
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
# Format with YAPF
uv run yapf -i -r src/

# Lint with Ruff
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

# Override specific parameters at runtime
uv run naics-embedder train training.learning_rate=1e-4 data_loader.batch_size=16
```

**Note:** The project now uses dynamic Structure-Aware Curriculum Learning instead of static
curriculum stages. Training difficulty adapts automatically based on the NAICS hierarchical
structure and training progress.

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

### 4. Debugging Training Issues

**Check hyperbolic validity:**

```python
from naics_embedder.text_model.hyperbolic import check_lorentz_manifold_validity

# After computing embeddings
is_valid = check_lorentz_manifold_validity(embeddings_hyp, curvature=1.0)
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

### 5. Working with Configuration

**Hydra Configuration System:**

The project uses Hydra for hierarchical configuration management.

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

### 6. GPU Memory Optimization

**Auto-detect optimal batch size:**

```bash
# Auto-detect GPU memory and suggest batch size
uv run naics-embedder tools gpu --auto

# Apply suggested configuration
uv run naics-embedder tools gpu --auto --apply
```

**Manual configuration:**

```bash
# For 24GB GPU
uv run naics-embedder tools gpu --gpu-memory 24

# For 80GB GPU
uv run naics-embedder tools gpu --gpu-memory 80
```

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

### 2. Curvature Clamping

Hyperbolic curvature is **clamped to safe range** `[0.1, 10.0]` to prevent numerical instability:

```python
c = torch.clamp(self.curvature, min=0.1, max=10.0)
```

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
- **Benefits:** Low memory footprint, fast random access
- **Format:** Parquet files with efficient columnar storage

### 5. Tokenization Caching

Tokenized inputs are **cached to disk** to avoid re-tokenization:

- **Implementation:** `text_model/dataloader/tokenization_cache.py`
- **Cache location:** `data/*.cache`
- **Invalidation:** Automatic on tokenizer or data changes

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

## Testing and Validation

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

**Workflow:** `.github/workflows/docs.yml`

- **Trigger:** Push to `main` or `master` branch
- **Action:** Build and deploy MkDocs documentation to GitHub Pages
- **Output:** <https://lowmason.github.io/naics-embedder/>

### Documentation

**Build locally:**

```bash
# Build docs
uv run mkdocs build

# Serve locally
uv run mkdocs serve
```

**View at:** <http://localhost:8000>

**Update API docs:**

API documentation is **auto-generated** from docstrings using `mkdocstrings`.

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

**Solutions:**

- Reduce learning rate
- Increase gradient clipping: `training.trainer.gradient_clip_val=0.5`
- Check curvature is clamped properly
- Enable mixed precision training

### 2. Low Hierarchy Correlation

**Symptom:** `hierarchy_corr` metric remains low (<0.3)

**Causes:**

- Insufficient training
- Loss weights not balanced
- Ground truth distances not informative

**Solutions:**

```bash
# Investigate ground truth distances
uv run naics-embedder tools investigate

# Increase hierarchy loss weight
uv run naics-embedder train loss.hierarchy_weight=0.5

# Train longer
uv run naics-embedder train training.trainer.max_epochs=30
```

### 3. OOM (Out of Memory) Errors

**Solutions:**

```bash
# Auto-optimize batch size
uv run naics-embedder tools gpu --auto --apply

# Manual reduction
uv run naics-embedder train data_loader.batch_size=8

# Increase gradient accumulation
uv run naics-embedder train training.trainer.accumulate_grad_batches=4
```

### 4. Checkpoint Not Found

**Symptom:** `FileNotFoundError: checkpoint not found`

**Solutions:**

```bash
# Check checkpoint directory exists
ls checkpoints/<experiment_name>/

# Use correct checkpoint path
uv run naics-embedder train --ckpt-path checkpoints/<experiment_name>/last.ckpt

# Or use "last" to auto-detect
uv run naics-embedder train --ckpt-path last
```

## Git Workflow

### Branch Naming

Follow the pattern: `claude/claude-md-<session-id>`

**Example:** `claude/claude-md-midviflvdkx1kn66-01PeC3NREbC6j5KUwMieyXiy`

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add hierarchy preservation loss to training loop"
git commit -m "Fix gradient flow issue in hyperbolic convolution"
git commit -m "Update curriculum 02_text with lower learning rate"

# Bad
git commit -m "fix bug"
git commit -m "updates"
```

### Push Workflow

Always push to the designated Claude branch:

```bash
# Push with upstream tracking
git push -u origin claude/claude-md-midviflvdkx1kn66-01PeC3NREbC6j5KUwMieyXiy

# If push fails due to network, retry with exponential backoff
# (handled automatically by git retry logic)
```

## File Modification Guidelines

### When to Edit vs. Create

**ALWAYS prefer editing** existing files over creating new ones:

- Modifying loss functions → Edit `text_model/loss.py`
- Adding CLI command → Edit `cli/commands/*.py`
- Adjusting config → Edit `conf/config.yaml` or curriculum files

**ONLY create new files** when:

- Adding entirely new module (e.g., new model architecture)
- Adding new data preprocessing script
- Creating new curriculum stage config

### Key Files to Edit

**Configuration:**

- `conf/config.yaml` - Base training configuration
- `conf/data_loader/*.yaml` - Data loading and tokenization configs

**Model Architecture:**

- `text_model/encoder.py` - Multi-channel encoder
- `text_model/moe.py` - Mixture-of-Experts
- `text_model/loss.py` - Loss functions
- `graph_model/hgcn.py` - Hyperbolic GCN

**Training:**

- `text_model/naics_model.py` - Main PyTorch Lightning module
- `cli/commands/training.py` - Training CLI commands

**Data:**

- `data/*.py` - Data preprocessing scripts

## Summary Checklist for AI Assistants

When working on this codebase:

- [ ] Use `uv run` for all CLI commands
- [ ] Follow single-quote Python style (`'` not `"`)
- [ ] Keep line length ≤ 100 characters
- [ ] Use semantic section dividers in Python files
- [ ] Add type hints to function signatures
- [ ] Use `logging` instead of `print`
- [ ] Test changes with a quick training run (`max_epochs=2`)
- [ ] Check hyperbolic validity when modifying geometry code
- [ ] Update configuration files (not hardcoded values) for hyperparameters
- [ ] Commit with descriptive messages
- [ ] Push to the designated Claude branch with `-u origin <branch-name>`

## Additional Resources

- **README.md:** High-level architecture overview
- **docs/usage.md:** Complete CLI command reference
- **docs/text_training.md:** Detailed text model training guide
- **docs/hgcn_training.md:** HGCN refinement guide
- **docs/literature.md:** Research background and citations
- **API Docs:** <https://lowmason.github.io/naics-embedder/>

## Contact and Support

- **Repository:** <https://github.com/lowmason/naics-embedder>
- **Author:** Lowell Mason
- **Issues:** Report bugs or request features via GitHub Issues
