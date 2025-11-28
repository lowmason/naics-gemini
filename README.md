# NAICS Hyperbolic Embedding System

<!-- markdownlint-disable MD013 -->
[![License](https://img.shields.io/github/license/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/blob/main/LICENSE)
[![Documentation](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml)
[![Tests](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml)
[![Issues](https://img.shields.io/github/issues/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/issues)
[![Last Commit](https://img.shields.io/github/last-commit/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/commits/main)
[![Contributors](https://img.shields.io/github/contributors/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/graphs/contributors)
[![Repo size](https://img.shields.io/github/repo-size/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder)
[![Top language](https://img.shields.io/github/languages/top/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder)

---

This project implements a unified hyperbolic representation learning framework for the **North American Industry Classification System (NAICS)**. The system combines multi-channel text encoding, Mixture-of-Experts fusion, hyperbolic contrastive learning, and a hyperbolic graph refinement stage to produce geometry-aware embeddings aligned with the hierarchical structure of the NAICS taxonomy.

The final output is a set of **Lorentz-model hyperbolic embeddings** suitable for similarity search, hierarchical modeling, graph-based reasoning, and downstream machine learning applications.

## 1. System Architecture Overview

The system consists of four sequential stages:

1. **Multi-channel text encoding** – independent transformer-based encoders for title, description, examples, and exclusions.
2. **Mixture-of-Experts (MoE) fusion** – adaptive fusion of the four embeddings using Top-2 gating.
3. **Hyperbolic contrastive learning** – projection into Lorentz space and optimization with Decoupled Contrastive Learning (DCL).
4. **Hyperbolic Graph Convolutional Refinement (HGCN)** – structure-aware refinement using the explicit NAICS parent–child graph.

Each stage is designed to preserve or enhance the hierarchical geometry of NAICS codes.

---

## 2. Stage 1 — Multi-Channel Text Encoding

Each NAICS code includes four distinct text fields:

- Title: Short code name (Concise category identification)
- Description: Detailed explanation of what the code encompasses (Rich semantic content)
- Examples: Representative businesses in this category (Concrete instantiations)
- Excluded: Codes explicitly NOT in this category (Disambiguation and boundaries)

Each field is processed independently using a transformer encoder (LoRA-adapted). This produces
four Euclidean embeddings:

- Title: (Embedding_title)
- Description: (Embedding_description)
- Examples: Embedding_examples)
- Excluded: (Embedding_excluded)

These embeddings serve as inputs to the fusion stage.

---

## 3. Stage 2 — Mixture-of-Experts Fusion (Top-2 Gating)

The four channel embeddings are concatenated and passed into a **Mixture-of-Experts (MoE)** module. Key components include:

- Top-2 gating to route each input to the two most relevant experts.
- Feed-forward expert networks that learn specialized fusion behaviors.
- Auxiliary load-balancing loss to ensure even expert utilization across batches.

This produces a single fused Euclidean embedding (E_fused) per NAICS code.

---

## 4. Stage 3 — Hyperbolic Contrastive Learning (Lorentz Model)

To align the latent space with the hierarchical structure of NAICS, embeddings are projected into **Lorentz-model hyperbolic space** via the exponential map.

### 4.1 Hyperbolic Projection

The fused Euclidean vector is mapped onto the hyperboloid:

- Uses exponential map at the origin
- Supports learned or fixed curvature
- Ensures numerical stability

The result is a Lorentz embedding (E_hyp).

### 4.2 Decoupled Contrastive Learning (DCL) Loss

Contrastive learning is performed using **Decoupled Contrastive Learning (DCL)** with
**Lorentzian geodesic distances**:

d(u, v) = arcosh(-<u, v>_L)

The DCL loss decouples the positive and negative terms:

L = (-pos_sim + logsumexp(neg_sims)).mean()

This formulation provides better gradient flow and numerical stability compared to standard InfoNCE.

Negatives include:

- unrelated codes,
- hierarchically distant codes,
- false negatives detected via periodic clustering (masked with -inf).

### 4.3 False Negative Mitigation

A curriculum-based procedure removes semantically similar negatives once the embedding space stabilizes:

1. Generate embeddings for the dataset.
2. Cluster embeddings (e.g., via KMeans).
3. Identify negatives sharing the cluster label with the anchor.
4. Exclude these from the contrastive denominator.

This prevents the model from incorrectly separating close hierarchical neighbors.

---

## 5. Stage 4 — Hyperbolic Graph Convolutional Refinement (HGCN)

To fully integrate the explicit hierarchical relationships of NAICS, the system applies a
**Hyperbolic Graph Convolutional Network** as a refinement stage.

### 5.1 Graph Structure

Nodes represent NAICS codes, and edges represent parent–child relationships in the taxonomy.

### 5.2 HGCN Layers

The refinement module includes:

- Two hyperbolic graph convolutional layers
- Tangent-space aggregation and message passing
- Learnable curvature shared across layers
- Exponential and logarithmic maps for manifold transitions

### 5.3 Refinement Objectives

The model optimizes a combined loss:

#### a. Hyperbolic Triplet Loss

Ensures that:

- anchor–positive distance < anchor–negative distance
- distances use Lorentz geodesics

#### b. Per-Level Radial Regularization

Encourages embeddings at the same hierarchical level to maintain similar hyperbolic radii.

This aligns global and local geometric structure with the NAICS taxonomy.

---

## 6. Final Output

Upon completion of all four stages, the system produces:

- High-fidelity hyperbolic embeddings in Lorentz space
- Representations consistent with both text semantics and hierarchical relationships
- Embeddings suitable for:
  - hierarchical search and retrieval
  - clustering and visualization
  - downstream machine learning tasks
  - graph-based analytics

---

This README provides a formal overview of the architecture, methodology, and geometric principles
underlying the NAICS embedding system. Further implementation details are available within the
project modules.

---

## 7. Architecture Diagram (Textual)

```text
+-------------------------------+
|  Multi-Channel Text Encoder   |
|  (Title / Desc / Examples /   |
|   Excluded via Transformer)   |
+---------------+---------------+
                |
                v
+-------------------------------+
|     Mixture-of-Experts        |
|  Top-2 Gating + Expert MLPs   |
|  Load-Balanced Fusion Layer   |
+---------------+---------------+
                |
                v
+-------------------------------+
|   Hyperbolic Projection       |
|   (Lorentz Exponential Map)   |
+---------------+---------------+
                |
                v
+-------------------------------+
| Hyperbolic Contrastive Loss   |
| (DCL + Lorentz Distance +    |
|  False Negative Masking)     |
+---------------+---------------+
                |
                v
+-------------------------------+
|          HGCN Refinement      |
|  (Tangent-Space GNN + Curv.)  |
+---------------+---------------+
                |
                v
+-------------------------------+
| Final Lorentz Hyperbolic Emb. |
+-------------------------------+
```

---

## 8. Onboarding Guide

### 8.0 Initial Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/lowmason/naics-embedder.git
   cd naics-embedder
   ```

2. Install uv:

   ```bash
   pip3 install uv
   ```

3. Install dependencies:

   ```bash
   uv synv
   ```

### 8.1 Download and preprocess NAICS data

1. Prepare the NAICS dataset with four text channels.

   ```bash
   uv run naics-embedder data preprocess
   uv run naics-embedder data relations
   uv run naics-embedder data distances
   uv run naics-embedder data triplets
   ```

   Or:

   ```bash
   uv run naics-embedder data all
   ```

### 8.2 Training the Contrastive Model

The text encoder now uses the Structure-Aware Dynamic Curriculum (SADC) scheduler by default. It
progresses through three phases in a single run—structural initialization, geometric refinement,
and false-negative mitigation—activating the appropriate sampling flags automatically.

Need a simpler curriculum for ablations? Set `curriculum.phase_mode=two_phase` to merge
Phase 3 into Phase 2, or enable the new annealing schedule from
[Issue #44](https://github.com/lowmason/naics-embedder/issues/44) via:

```yaml
curriculum:
  anneal:
    enabled: true
    alpha_start: 1.5
    alpha_end: 0.8
    epochs: 40
```

This gradually reduces the tree-distance exponent and increases router-guided mixing, and
can optionally be triggered by the logged `adaptive_margin_mean` metric.

Run training with your base config and optional overrides:

```bash
uv run naics-embedder train --config conf/config.yaml \
  training.learning_rate=1e-4 training.trainer.max_epochs=15
```

To compare against the static SANS baseline proposed in
[Issue #43](https://github.com/lowmason/naics-embedder/issues/43), switch the data-layer
sampler via:

```bash
uv run naics-embedder train sampling.strategy=sans_static \
  sampling.sans_static.near_bucket_weight=0.7
```

This keeps the model-side curriculum intact while replacing Phase 1 tree-distance weighting
with fixed near/far probabilities.

To experiment with alternative false-negative treatments from
[Issue #45](https://github.com/lowmason/naics-embedder/issues/45), tweak the new block:

```yaml
false_negatives:
  strategy: hybrid
  attraction_weight: 0.2
  attraction_metric: l2
```

`eliminate` (default) masks suspected false negatives, `attract` keeps them but adds an
auxiliary attraction loss, and `hybrid` does both.

Key flags managed by SADC during training:

- Tree-distance weighting and sibling masking during structural initialization
- Router-guided and hard-negative mining during geometric refinement
- Clustering-driven false-negative elimination in the final phase

**Migrating from legacy stage files:** The old multi-file curriculum chains are retired. Use the
single `train` command with overrides instead of stage-by-stage invocations. The deprecated
`train-seq --legacy` remains for backward compatibility but is no longer required for SADC.

### 8.3 Running HGCN Refinement

1. Construct the NAICS parent–child graph.
2. Load hyperbolic embeddings from 8.1.2.
3. Train the refinement model:

   ```bash
   uv run naics-embedder train-hgcn --config configs/hgcn.yaml
   ```

4. Save refined embeddings for downstream tasks.

---

## 9. Using the Final Embeddings

### 9.1 Similarity Search

Use Lorentzian distance:

```python
dist = lorentz_distance(x, y)
```

Lower values indicate closer hierarchical or semantic similarity.

### 9.2 Visualization

Project to tangent space or Poincaré ball for plotting.

### 9.3 Downstream ML

Final embeddings can be used as features for:

- classification models,
- clustering algorithms (in hyperbolic or tangent space),
- retrieval and recommendation systems.

---
