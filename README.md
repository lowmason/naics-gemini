# NAICS Gemini

**Hierarchical Contrastive Learning for NAICS Industry Classification Codes**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Learn semantically rich, hierarchy-preserving embeddings for 2,125 NAICS codes using multi-channel contrastive learning in hyperbolic space.

---

## Overview

NAICS Gemini is a production-ready training system that learns hierarchical embeddings for North American Industry Classification System (NAICS) codes. The system combines:

- 🔤 **Multi-Channel Encoding**: Separate LoRA-tuned transformers for title, description, exclusions, and examples
- 🧠 **Mixture of Experts**: Dynamic routing through 4 specialized fusion experts
- 📐 **Hyperbolic Geometry**: Lorentz model for natural hierarchy representation
- 📚 **Curriculum Learning**: Progressive difficulty from easy (sectors) to hard (siblings, exclusions)
- ⚡ **Efficient Training**: Streaming 263M+ triplets with curriculum-based filtering

### Key Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Structure** | Preserves 6-level NAICS taxonomy (sectors → national industries) |
| **Multi-Channel Fusion** | Learns to combine 4 text sources dynamically per code |
| **Hyperbolic Space** | Lorentz model naturally represents tree structures |
| **Curriculum Learning** | 8 hardness levels from unrelated (93%) to exclusions (0.05%) |
| **Parameter Efficient** | LoRA fine-tuning: ~1% of full transformer parameters |
| **Scalable** | Streaming dataset handles 263M triplets without loading into RAM |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM) or Apple Silicon (16GB+ unified memory)
- 10GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/lowmason/naics-gemini.git
cd naics-gemini

# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install -e .
```

### Generate Training Data

```bash
# Complete pipeline (~20 minutes)
uv run naics-gemini data all

# Or run stages individually
uv run naics-gemini data preprocess  # Download & clean NAICS data
uv run naics-gemini data distances   # Compute graph distances
uv run naics-gemini data triplets    # Generate training triplets
```

This creates three datasets:
- `data/naics_descriptions.parquet` (1.2 MB, 2,125 codes)
- `data/naics_distances.parquet` (48 MB, 3M pairs)
- `data/naics_training_pairs.parquet` (3.2 GB, 263M triplets)

### Train Your First Model

```bash
# Easy curriculum: Learn sector boundaries (2-4 hours on RTX 4090)
uv run naics-gemini train --curriculum 01_stage_easy

# Monitor training
tensorboard --logdir outputs/01_stage_easy
```

**That's it!** Your model checkpoint will be saved to `checkpoints/01_stage_easy/`.

For detailed instructions, see **[Quick Start Guide →](docs/quickstart.md)**

---

## Documentation

### 📚 Complete Documentation Suite

| Document | Description | When to Read |
|----------|-------------|--------------|
| **[Quick Start Guide](docs/quickstart.md)** | Installation, data pipeline, first training run | Start here! |
| **[Curriculum Design Guide](docs/curriculum_design_guide.md)** | Understanding hardness, creating custom curricula | Before training experiments |
| **[Dataset Specification](docs/dataset_specification.md)** | Schemas, validation rules, usage patterns | When working with data |
| **[Architecture Guide](docs/architecture.md)** | System design, model components, performance | Understanding how it works |
| **[Troubleshooting Guide](docs/troubleshooting.md)** | Debugging training, fixing errors | When things go wrong |

### 🎯 Quick Navigation

**I want to...**
- 🚀 **Get started**: [Quick Start Guide](docs/quickstart.md)
- 📖 **Understand hardness levels**: [Curriculum Design → Understanding Hardness](docs/curriculum_design_guide.md#understanding-hardness-levels)
- 🎨 **Create custom curriculum**: [Curriculum Design → Creating Custom](docs/curriculum_design_guide.md#creating-custom-curricula)
- 📊 **Validate my data**: [Dataset Specification → Validation](docs/dataset_specification.md#validation-script)
- 🐛 **Fix NaN loss**: [Troubleshooting → Loss is NaN](docs/troubleshooting.md#issue-loss-is-nan)
- ⚡ **Speed up training**: [Troubleshooting → Performance](docs/troubleshooting.md#performance-issues)
- 🏗️ **Understand architecture**: [Architecture Guide](docs/architecture.md)

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                             │
│  Census Bureau → Preprocess → Distances → Triplets          │
│  (4 Excel files)    (2,125)     (3.0M)      (263M)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 MULTI-CHANNEL ENCODER                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │  Title  │  │  Desc   │  │Excluded │  │Examples │       │
│  │ + LoRA  │  │ + LoRA  │  │ + LoRA  │  │ + LoRA  │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       └────────────┴─────────────┴────────────┘             │
│                         ↓                                    │
│                 MIXTURE OF EXPERTS                           │
│          (4 experts, Top-2 gating)                          │
│                         ↓                                    │
│            HYPERBOLIC PROJECTION                             │
│              (Lorentz Model)                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
                  CONTRASTIVE LOSS
              (InfoNCE in Hyperbolic Space)
```

### Key Components

1. **Multi-Channel Encoder** ([details](docs/architecture.md#multi-channel-encoder))
   - 4 separate LoRA-tuned transformers (all-mpnet-base-v2)
   - Each channel: title, description, excluded, examples
   - LoRA config: r=8, alpha=16, target=['query', 'value']

2. **Mixture of Experts** ([details](docs/architecture.md#architecture-multi-channel-fusion))
   - 4 expert networks with Top-2 gating
   - Learns specialized fusion strategies
   - Global-batch load balancing (λ=0.01)

3. **Hyperbolic Projection** ([details](docs/architecture.md#hyperbolic-projection))
   - Exponential map to Lorentz hyperboloid
   - Curvature: c=1.0
   - Stable gradients (no NaN issues)

4. **Training** ([details](docs/architecture.md#training-architecture))
   - Curriculum-based triplet sampling
   - Streaming dataset (no RAM overload)
   - AdamW optimizer + Cosine schedule

See **[Architecture Guide](docs/architecture.md)** for complete details.

---

## Training

### Built-in Curricula

| Stage | Difficulty | Positive Pairs | Negative Buckets | Duration |
|-------|-----------|----------------|------------------|----------|
| **01_easy** | Easy | Level 6, dist≤2.0 | [1, 2] Unrelated + Distant | 2-4 hours |
| **02_medium** | Medium | Levels 5-6, dist≤3.0 | [1, 2, 5, 6] Mixed | 4-6 hours |
| **03_hard** | Hard | All levels | [1, 2, 3, 4, 5, 6, 8] Full | 8-12 hours |

### Training Commands

```bash
# Stage 1: Learn sector boundaries
uv run naics-gemini train --curriculum 01_stage_easy

# Stage 2: Within-sector distinctions
uv run naics-gemini train --curriculum 02_stage_medium

# Stage 3: Fine-grained + exclusions
uv run naics-gemini train --curriculum 03_stage_hard

# Custom overrides
uv run naics-gemini train -c 01_stage_easy \
  training.trainer.max_epochs=10 \
  data.batch_size=64 \
  model.lora.r=16
```

### Understanding Hardness Levels

Training triplets are labeled 1 (easiest) to 8 (hardest):

| Level | % of Data | Description | Distance Diff |
|-------|-----------|-------------|---------------|
| 1 | 93.36% | **Unrelated** - Different sectors | 7.0 |
| 2 | 0.14% | **Distant** - Same sector, far apart | 4.5-6.5 |
| 3 | 0.15% | **Moderate** - Related subsectors | 3.5-4.0 |
| 4 | 0.31% | **Cousins** - Related industries | 2.5-3.0 |
| 5 | 1.37% | **Close** - Close relatives | 2.0 |
| 6 | 4.61% | **Siblings** - Same parent | 1.0 |
| 7 | 0.00% | **Very Hard** - Parent-child | 0.5 |
| 8 | 0.05% | **Exclusions** - Semantically close but excluded | Variable |

See **[Curriculum Design Guide](docs/curriculum_design_guide.md)** for creating custom curricula.

---

## Dataset Details

### NAICS Descriptions (`naics_descriptions.parquet`)

Complete text data for all 2,125 NAICS codes:

```python
Schema([
    ('index', UInt32),           # Unique ID (0-2124)
    ('level', UInt8),            # Hierarchy depth (2-6)
    ('code', String),            # NAICS code (e.g., "541511")
    ('title', String),           # Official title
    ('description', String),     # Detailed description
    ('examples', String),        # Illustrative examples (nullable)
    ('excluded', String),        # Exclusion text (nullable)
    ('excluded_codes', List)     # Parsed exclusion codes (nullable)
])
```

**Size:** 1.2 MB | **Rows:** 2,125

### NAICS Distances (`naics_distances.parquet`)

Pairwise graph distances for all code pairs:

```python
Schema([
    ('idx_i', UInt32),      # Index of first code
    ('idx_j', UInt32),      # Index of second code
    ('code_i', String),     # First NAICS code
    ('code_j', String),     # Second NAICS code
    ('distance', Float32)   # Graph distance (0.5-10.0)
])
```

**Size:** 48 MB | **Rows:** 3,004,420

### NAICS Training Pairs (`naics_training_pairs.parquet`)

Training triplets with hardness annotations:

```python
Schema([
    ('anchor_code', String),
    ('positive_code', String),
    ('negative_code', String),
    ('excluded', Boolean),          # Is hardness 8?
    ('unrelated', Boolean),         # Is hardness 1?
    ('positive_distance', Float32),
    ('negative_distance', Float32),
    ('distance_diff', Float32)      # Difficulty metric
])
```

**Size:** 3.2 GB | **Rows:** 263,830,364

See **[Dataset Specification](docs/dataset_specification.md)** for complete schemas, validation rules, and usage patterns.

---

## Configuration

All configuration via Hydra YAML files in `conf/`:

```
conf/
├── config.yaml              # Base configuration
├── curriculum/              # Training curricula
│   ├── 01_stage_easy.yaml
│   ├── 02_stage_medium.yaml
│   └── 03_stage_hard.yaml
├── data/
│   └── default.yaml         # Data loading config
├── model/
│   └── default.yaml         # Model architecture
├── loss/
│   └── default.yaml         # Loss function settings
└── training/
    └── default.yaml         # Optimizer & trainer
```

### Example: Custom Curriculum

```yaml
# conf/curriculum/my_experiment.yaml
name: my_experiment

# Positive filtering
positive_levels: [6]
positive_distance_max: 2.0

# Negative sampling
difficulty_buckets: [1, 5, 6]
bucket_percentages:
  1: 0.50   # 50% unrelated
  5: 0.30   # 30% close relatives
  6: 0.20   # 20% siblings

k_negatives: 16
```

Train with: `uv run naics-gemini train -c my_experiment`

---

## Results

### Expected Performance

| Curriculum | Val Loss | Embedding Quality | Training Time (RTX 4090) |
|-----------|----------|-------------------|-------------------------|
| 01_easy | 2.0-2.5 | Sector separation | 2-4 hours |
| 02_medium | 1.5-2.0 | Subsector structure | 4-6 hours |
| 03_hard | 1.0-1.5 | Fine-grained + exclusions | 8-12 hours |

### Evaluation Metrics

- **Contrastive Loss**: Should decrease steadily
- **Load Balancing**: Should stay < 0.05 (MoE not collapsed)
- **Hierarchy Preservation**: Check with Cophenetic correlation
- **Embedding Space**: Verify sibling proximity, sector separation

---

## Troubleshooting

### Common Issues

| Issue | Quick Fix | Details |
|-------|-----------|---------|
| 💥 **NaN loss** | Reduce LR: `training.learning_rate=5e-5` | [Link](docs/troubleshooting.md#issue-loss-is-nan) |
| 📈 **Loss not decreasing** | Easier curriculum: `difficulty_buckets: [1]` | [Link](docs/troubleshooting.md#issue-loss-not-decreasing) |
| 💾 **Out of memory** | Reduce batch: `data.batch_size=16` | [Link](docs/troubleshooting.md#issue-out-of-memory-oom) |
| 🐌 **Slow training** | More workers: `data.num_workers=8` | [Link](docs/troubleshooting.md#issue-training-extremely-slow) |
| 🔄 **MoE collapse** | Increase λ: `model.moe.load_balancing_coef=0.02` | [Link](docs/troubleshooting.md#issue-moe-mode-collapse) |

See **[Troubleshooting Guide](docs/troubleshooting.md)** for complete debugging procedures.

---

## Project Structure

```
naics-gemini/
├── conf/                    # Hydra configuration files
│   ├── config.yaml
│   ├── curriculum/
│   ├── data/
│   ├── model/
│   ├── loss/
│   └── training/
├── data/                    # Generated datasets (gitignored)
│   ├── naics_descriptions.parquet
│   ├── naics_distances.parquet
│   └── naics_training_pairs.parquet
├── docs/                    # Documentation
│   ├── quickstart.md
│   ├── curriculum_design_guide.md
│   ├── dataset_specification.md
│   ├── architecture.md
│   └── troubleshooting.md
├── src/naics_gemini/        # Source code
│   ├── data/                # Data loading & streaming
│   ├── data_generation/     # Pipeline for creating datasets
│   ├── model/               # Neural network components
│   └── utils/               # Utilities (hyperbolic, logging)
├── checkpoints/             # Model checkpoints (gitignored)
├── outputs/                 # Training logs (gitignored)
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

---

## Requirements

### Core Dependencies

- **PyTorch** 2.5.1 (CUDA 12.1 or MPS)
- **PyTorch Lightning** 2.4+
- **Transformers** 4.46+ (HuggingFace)
- **PEFT** 0.13+ (LoRA)
- **Polars** 1.9+ (Data processing)
- **Hydra** 1.3+ (Configuration)

See `pyproject.toml` for complete list.

### Hardware Recommendations

| Setup | Batch Size | K Negatives | Training Speed |
|-------|-----------|-------------|----------------|
| **Minimum** | 16 | 8 | 8-10 hours |
| RTX 3090 (24GB) | 32 | 16 | 4-6 hours |
| RTX 4090 (24GB) | 32 | 16 | 2-4 hours |
| A100 (40GB) | 64 | 32 | 1-2 hours |
| Apple M1/M2 Max | 16 | 8 | 10-12 hours |

---

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
pytest tests/

# Run linting
ruff check src/
ruff format src/
```

### Code Style

- **Formatter**: Ruff (single quotes, 100 char lines)
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

See `pyproject.toml` for complete style configuration.

---

## Citation

If you use NAICS Gemini in your research, please cite:

```bibtex
@software{naics_gemini2025,
  title={NAICS Gemini: Hierarchical Contrastive Learning for Industry Classification},
  author={[Your Name]},
  year={2025},
  url={https://github.com/lowmason/naics-gemini}
}
```

---

## Related Work

### Hyperbolic Embeddings
- Nickel & Kiela (2017) - [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)
- Nickel & Kiela (2018) - [Learning Continuous Hierarchies in the Lorentz Model](https://arxiv.org/abs/1806.03417)

### Contrastive Learning
- Chen et al. (2020) - [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- Khosla et al. (2020) - [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)

### Mixture of Experts
- Shazeer et al. (2017) - [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- Lepikhin et al. (2020) - [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **NAICS Data**: U.S. Census Bureau
- **Base Model**: [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **Framework**: PyTorch Lightning, Hydra, PEFT

---

## Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/lowmason/naics-gemini/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/lowmason/naics-gemini/discussions)

---

**Built with ❤️ for hierarchical representation learning**
