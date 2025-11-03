# NAICS Gemini Documentation

Comprehensive documentation for the NAICS Gemini hierarchical contrastive learning system.

## 📚 Documentation Index

### Getting Started
- **[Quick Start Guide](quickstart.md)** - Installation, data pipeline, and first training run
  - Prerequisites and installation
  - Running the data pipeline (preprocess → distances → triplets)
  - Training your first model
  - Monitoring and evaluation
  - Common issues and solutions

### Core Concepts
- **[Curriculum Design Guide](curriculum_design_guide.md)** - Designing effective training curricula
  - Understanding hardness levels (1-8)
  - Configuration parameters explained
  - Design principles and best practices
  - Built-in curricula walkthrough
  - Creating custom curricula
  - Advanced strategies

### Technical Deep-Dive
- **[Architecture Guide](architecture.md)** - System design and implementation
  - Data pipeline architecture
  - Model architecture (Multi-Channel Encoder, MoE, Hyperbolic Space)
  - Training architecture
  - Key design decisions and rationale
  - Performance characteristics

### Problem Solving
- **[Troubleshooting Guide](troubleshooting.md)** - Debugging and issue resolution
  - Data pipeline issues
  - Training issues (NaN loss, convergence, etc.)
  - Model architecture issues (MoE collapse, LoRA)
  - Performance issues (OOM, slow training)
  - Curriculum issues
  - Environment issues

## 🎯 Quick Navigation

**I want to...**

| Goal | Document | Section |
|------|----------|---------|
| Install and run my first training | [Quick Start](quickstart.md) | Installation → Training |
| Understand hardness levels | [Curriculum Design](curriculum_design_guide.md) | Understanding Hardness Levels |
| Create a custom curriculum | [Curriculum Design](curriculum_design_guide.md) | Creating Custom Curricula |
| Understand how the model works | [Architecture](architecture.md) | Model Architecture |
| Fix NaN loss | [Troubleshooting](troubleshooting.md) | Training Issues → Loss is NaN |
| Speed up training | [Troubleshooting](troubleshooting.md) | Performance Issues |
| Understand MoE architecture | [Architecture](architecture.md) | Model Architecture → Multi-Channel Encoder |
| Learn about hyperbolic space | [Architecture](architecture.md) | Hyperbolic Projection |
| Fix OOM errors | [Troubleshooting](troubleshooting.md) | Performance Issues → Out of Memory |
| Understand data pipeline | [Architecture](architecture.md) | Data Pipeline Architecture |

## 📖 Recommended Reading Order

### For New Users
1. [Quick Start Guide](quickstart.md) - Get up and running
2. [Curriculum Design Guide](curriculum_design_guide.md) - Understand training strategy
3. [Architecture Guide](architecture.md) - Learn how it works (optional but recommended)

### For Developers
1. [Architecture Guide](architecture.md) - Understand system design
2. [Curriculum Design Guide](curriculum_design_guide.md) - See how curricula affect training
3. [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

### For Troubleshooting
1. [Troubleshooting Guide](troubleshooting.md) - Start here for specific issues
2. [Architecture Guide](architecture.md) - Understand root causes
3. [Quick Start Guide](quickstart.md#common-issues) - Quick fixes

## 🔑 Key Concepts

### Hardness Levels
Training triplets are labeled with hardness from 1 (easiest) to 8 (hardest):
- **Level 1 (93%):** Unrelated codes (different sectors)
- **Level 6 (4.6%):** Siblings (same parent)
- **Level 8 (0.05%):** Exclusions (semantically close but explicitly different)

### Curriculum Learning
Progressive difficulty: start with easy distinctions (sectors), gradually introduce hard negatives (siblings, exclusions).

### Hyperbolic Space
Embeddings live in Lorentz hyperboloid model for natural hierarchy representation (exponential volume growth).

### Mixture of Experts (MoE)
4 specialized fusion strategies (experts) with Top-2 gating dynamically combine 4 text channels.

## 📊 Dataset Overview

| Dataset | Rows | Size | Description |
|---------|------|------|-------------|
| `naics_descriptions.parquet` | 2,125 | 1.2 MB | NAICS codes with text fields |
| `naics_distances.parquet` | 3.0M | 48 MB | Pairwise graph distances |
| `naics_training_pairs.parquet` | 263M | 3.2 GB | Training triplets with hardness |

## 🛠️ System Components

```
Data Pipeline → Multi-Channel Encoder → MoE Fusion → Hyperbolic Projection → Contrastive Loss
```

- **Data Pipeline:** Downloads, cleans, and structures NAICS data
- **Multi-Channel Encoder:** 4 LoRA-tuned transformers (title, description, excluded, examples)
- **MoE Fusion:** Routes channels through specialized experts
- **Hyperbolic Projection:** Maps to Lorentz model for hierarchy preservation
- **Contrastive Loss:** Pulls positives together, pushes negatives apart

## ⚡ Quick Tips

**First time training:**
```bash
uv run naics-gemini data all          # Generate datasets (~20 min)
uv run naics-gemini train -c 01_stage_easy  # Train easy curriculum (~3 hours)
```

**Creating custom curriculum:**
```bash
cp conf/curriculum/01_stage_easy.yaml conf/curriculum/my_experiment.yaml
# Edit my_experiment.yaml
uv run naics-gemini train -c my_experiment
```

**Debugging slow training:**
```bash
# Increase data workers
uv run naics-gemini train -c 01_stage_easy data.num_workers=8

# Reduce batch size if OOM
uv run naics-gemini train -c 01_stage_easy data.batch_size=16
```

## 🤝 Contributing

Found an issue or have a suggestion? Please:
1. Check [Troubleshooting Guide](troubleshooting.md) first
2. Search existing GitHub issues
3. Open a new issue with details

## 📄 Additional Resources

- **GitHub Repository:** https://github.com/lowmason/naics-gemini
- **Research Paper:** [NAICS_Contrastive_Learning_System_Design.pdf](../NAICS_Contrastive_Learning_System_Design.pdf)
- **Program Outputs:** [Program_output_1.pdf](../Program_output_1.pdf)

## 📝 Document Stats

- **Quick Start:** 9.4 KB, ~2,000 words
- **Curriculum Design:** 19 KB, ~4,500 words
- **Architecture:** 36 KB, ~7,000 words
- **Troubleshooting:** 23 KB, ~5,000 words
- **Total:** ~87 KB, ~18,500 words

---

**Last Updated:** November 2025

**Version:** 0.1.0

**Maintainer:** NAICS Gemini Team
