# System Architecture

This document provides a comprehensive overview of the NAICS Hyperbolic Embedding System architecture, detailing each component, data flow, and design decisions.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Data Flow](#data-flow)
4. [Training Pipeline](#training-pipeline)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Design Decisions](#design-decisions)

---

## System Overview

The NAICS Hyperbolic Embedding System is a unified framework for learning hierarchical representations of the North American Industry Classification System (NAICS) taxonomy. The system combines:

- **Multi-channel text encoding** using transformer-based encoders
- **Mixture-of-Experts (MoE) fusion** for adaptive feature combination
- **Hyperbolic contrastive learning** in Lorentz space
- **False-negative mitigation** via curriculum clustering
- **Hierarchy preservation** through specialized loss functions
- **Optional HGCN refinement** for graph-based structure integration

The final output is a set of **Lorentz-model hyperbolic embeddings** that preserve the hierarchical structure of the NAICS taxonomy while capturing semantic relationships from text descriptions.

---

## Architecture Components

### 1. Multi-Channel Text Encoder

The system processes each NAICS code through four independent text channels:

- **Title**: Short code name (e.g., "Software Publishers")
- **Description**: Detailed explanation of the code
- **Examples**: Representative examples of businesses in this category
- **Excluded**: Codes explicitly excluded from this category

#### Implementation

```python
MultiChannelEncoder(
    base_model_name='sentence-transformers/all-mpnet-base-v2',
    lora_r=8,
    lora_alpha=16,
    num_experts=4,
    curvature=1.0
)
```

**Key Features:**

- **LoRA Adaptation**: Each channel uses a separate LoRA-adapted transformer encoder
  - Reduces trainable parameters while maintaining expressiveness
  - Universal LoRA targeting (`target_modules='all-linear'`) works with any transformer architecture
  - Default: `r=8`, `alpha=16`, `dropout=0.1`

- **Gradient Checkpointing**: Enabled by default to reduce memory usage
  - Trades computation for memory during backpropagation
  - Critical for large batch sizes or limited GPU memory

- **Channel Independence**: Each channel learns specialized representations
  - Title encoder focuses on concise category names
  - Description encoder captures detailed semantics
  - Examples encoder learns from representative instances
  - Excluded encoder learns negative semantics

**Output**: Four Euclidean embeddings `(E_title, E_desc, E_examples, E_excluded)`, each of dimension `embedding_dim` (typically 768 for MPNet).

---

### 2. Mixture-of-Experts (MoE) Fusion

The four channel embeddings are concatenated and passed through a Mixture-of-Experts layer for adaptive fusion.

#### Architecture

```python
MixtureOfExperts(
    input_dim=embedding_dim * 4,  # 4 channels concatenated
    hidden_dim=1024,
    num_experts=4,
    top_k=2
)
```

**Components:**

1. **Gating Network**: Linear layer that computes expert selection scores
   - Input: Concatenated channel embeddings `(batch_size, embedding_dim * 4)`
   - Output: Expert scores `(batch_size, num_experts)`

2. **Top-K Selection**: Selects the `top_k=2` most relevant experts per input
   - Reduces computation while maintaining expressiveness
   - Softmax normalization over selected experts

3. **Expert Networks**: Each expert is a 2-layer MLP:

   ```
   Linear(input_dim → hidden_dim) → ReLU → Dropout(0.1) → Linear(hidden_dim → input_dim)
   ```

4. **Load Balancing**: Auxiliary loss ensures even expert utilization
   - Prevents expert collapse (all inputs routed to same expert)
   - Coefficient: `load_balancing_coef=0.01`

**Output**: Fused Euclidean embedding `E_fused` of dimension `embedding_dim`, projected back from `embedding_dim * 4` via a linear projection layer.

---

### 3. Hyperbolic Projection

The fused Euclidean embedding is projected into **Lorentz-model hyperbolic space** to align with hierarchical structure.

#### Lorentz Model

The Lorentz model represents hyperbolic space as points on a hyperboloid:

- **Coordinates**: `(x₀, x₁, ..., xₙ)` where:
  - `x₀` is the time coordinate (hyperbolic radius)
  - `x₁...xₙ` are spatial coordinates
- **Constraint**: `-x₀² + x₁² + ... + xₙ² = -1/c` (Lorentz inner product)
- **Curvature**: `c` controls the curvature of the space (default: `c=1.0`)

#### Implementation

```python
HyperbolicProjection(
    input_dim=embedding_dim,
    curvature=1.0
)
```

**Projection Process:**

1. **Linear Projection**: Maps Euclidean embedding to tangent space
   - `Linear(embedding_dim → embedding_dim + 1)`
   - Adds the time coordinate dimension

2. **Exponential Map**: Maps from tangent space to hyperboloid

   ```
   x₀ = cosh(||v|| / √c)
   x_rest = (sinh(||v|| / √c) * v) / ||v||
   ```

   - Ensures points satisfy the Lorentz constraint
   - Numerically stable with clamping

**Output**: Hyperbolic embedding `E_hyp` of shape `(batch_size, embedding_dim + 1)` on the Lorentz hyperboloid.

---

### 4. Hyperbolic Contrastive Learning

Contrastive learning is performed directly in hyperbolic space using **Lorentzian geodesic distances**.

#### Hyperbolic InfoNCE Loss

```python
HyperbolicInfoNCELoss(
    embedding_dim=embedding_dim,
    temperature=0.07,
    curvature=1.0
)
```

**Distance Computation:**

Lorentzian distance between two points `u, v` on the hyperboloid:

```
d(u, v) = √c * arccosh(-⟨u, v⟩_L)
```

where the Lorentz inner product is:

```
⟨u, v⟩_L = u₁v₁ + ... + uₙvₙ - u₀v₀
```

**Loss Function:**

Standard InfoNCE loss with hyperbolic distances:

```
L = -log(exp(-d(anchor, positive) / τ) / Σᵢ exp(-d(anchor, negativeᵢ) / τ))
```

where `τ` is the temperature parameter (default: `0.07`).

**False-Negative Mitigation:**

A curriculum-based procedure removes semantically similar negatives:

1. **Clustering**: Periodically cluster embeddings using KMeans (default: every 5 epochs after epoch 10)
2. **Masking**: Identify negatives sharing cluster label with anchor
3. **Exclusion**: Mask these false negatives from the contrastive denominator

This prevents the model from incorrectly separating close hierarchical neighbors.

---

### 5. Hierarchy Preservation Loss

Additional loss component that directly optimizes hierarchy preservation by matching embedding distances to tree distances.

#### Implementation

```python
HierarchyPreservationLoss(
    tree_distances=ground_truth_distances,
    code_to_idx=code_to_idx,
    weight=0.325,  # Default weight
    min_distance=0.1
)
```

**Loss Computation:**

For each pair of codes in the batch:

```
L_hierarchy = weight * MSE(embedding_distance, tree_distance)
```

- **Embedding Distance**: Lorentzian geodesic distance between hyperbolic embeddings
- **Tree Distance**: Ground truth distance in the NAICS taxonomy tree
- **Weight**: Controls the importance of hierarchy preservation (default: `0.325`)

This loss encourages the embedding space to directly reflect the hierarchical structure of NAICS.

---

### 6. Rank Order Preservation Loss (LambdaRank)

Global ranking optimization using LambdaRank to preserve rank order relationships.

#### Implementation

```python
LambdaRankLoss(
    tree_distances=ground_truth_distances,
    code_to_idx=code_to_idx,
    weight=0.275,  # Default weight
    sigma=1.0,
    ndcg_k=10
)
```

**Key Features:**

- **Position-Aware**: Optimizes NDCG@k (Normalized Discounted Cumulative Gain)
- **Gradient Weighting**: Uses LambdaRank gradients that weight pairs by their impact on ranking
- **Global Optimization**: Considers all pairs, not just anchor-positive-negative triplets

This loss ensures that the relative ordering of codes in the embedding space matches their hierarchical relationships.

---

### 7. Radius Regularization

Prevents hyperbolic embeddings from collapsing to the origin or expanding too far.

#### Implementation

Regularization term that encourages embeddings to maintain reasonable hyperbolic radii:

```
L_radius = radius_reg_weight * ||r - target_radius||²
```

where `r` is the hyperbolic radius (time coordinate `x₀`) and `target_radius` is a learned or fixed value.

**Default Weight**: `0.01`

---

### 8. Evaluation Metrics

The system computes comprehensive evaluation metrics during training:

#### Hierarchy Metrics

- **Cophenetic Correlation**: Measures how well embedding distances preserve tree structure
- **Spearman Correlation**: Rank-order correlation between embedding and tree distances
- **NDCG@k**: Position-aware ranking quality metric (k ∈ {5, 10, 20})
- **Distortion**: Mean, median, and std deviation of distance distortions

#### Embedding Statistics

- **Lorentz Norm**: Mean and violations of the Lorentz constraint
- **Hyperbolic Radius**: Mean and std of hyperbolic radii
- **Pairwise Distances**: Mean and std of embedding distances

#### Collapse Detection

- **Norm CV**: Coefficient of variation of embedding norms
- **Distance CV**: Coefficient of variation of pairwise distances
- **Variance Collapse**: Detects if embeddings collapse to a single point

---

## Data Flow

### Forward Pass

```
NAICS Code (4 text channels)
    ↓
[Multi-Channel Encoder]
    ├─→ Title Encoder (LoRA) → E_title
    ├─→ Description Encoder (LoRA) → E_desc
    ├─→ Examples Encoder (LoRA) → E_examples
    └─→ Excluded Encoder (LoRA) → E_excluded
    ↓
[Concatenate] → (embedding_dim * 4)
    ↓
[MoE Fusion] → E_fused (embedding_dim)
    ↓
[Hyperbolic Projection] → E_hyp (embedding_dim + 1)
    ↓
[Lorentz Hyperboloid]
```

### Training Step

```
Batch: (anchors, positives, negatives)
    ↓
[Forward Pass] → Hyperbolic embeddings
    ↓
[Compute Distances]
    ├─→ Anchor-Positive distances
    └─→ Anchor-Negative distances
    ↓
[Apply False-Negative Mask] (if available)
    ↓
[Hyperbolic InfoNCE Loss]
    ↓
[Additional Losses]
    ├─→ Hierarchy Preservation Loss
    ├─→ LambdaRank Loss
    ├─→ Radius Regularization
    └─→ MoE Load Balancing Loss
    ↓
[Total Loss] → Backpropagation
```

---

## Training Pipeline

### Single-Stage Training

1. **Data Loading**: Stream triplets (anchor, positive, negatives) based on curriculum config
2. **Forward Pass**: Encode and project to hyperbolic space
3. **Loss Computation**: Combine contrastive, hierarchy, and ranking losses
4. **Backpropagation**: Update encoder, MoE, and projection parameters
5. **Evaluation**: Compute metrics every N epochs
6. **Early Stopping**: Monitor validation loss with patience

### Sequential Curriculum Training

Multiple stages with progressive difficulty:

- **Stage 1**: Coarse-grained relationships (e.g., level 2-3 codes)
- **Stage 2**: Finer-grained relationships (e.g., level 3-4 codes)
- **Stage 3+**: Specialized relationships or edge cases

Each stage:

1. Loads checkpoint from previous stage
2. Trains with stage-specific curriculum
3. Saves best checkpoint for next stage

### False-Negative Curriculum

After initial training (default: epoch 10), periodically:

1. Generate embeddings for all codes
2. Cluster embeddings using KMeans (default: 500 clusters)
3. Update pseudo-labels based on cluster assignments
4. Mask false negatives in subsequent training

---

## Mathematical Foundations

### Hyperbolic Geometry

The system uses the **Lorentz model** of hyperbolic space, which has several advantages:

1. **Differentiable**: Smooth operations suitable for gradient-based optimization
2. **Numerically Stable**: Well-conditioned distance computations
3. **Hierarchical Structure**: Natural representation for tree-like data

#### Lorentz Inner Product

For two points `u = (u₀, u₁, ..., uₙ)` and `v = (v₀, v₁, ..., vₙ)`:

```
⟨u, v⟩_L = Σᵢ₌₁ⁿ uᵢvᵢ - u₀v₀
```

#### Lorentz Distance

Geodesic distance on the hyperboloid:

```
d(u, v) = √c * arccosh(-⟨u, v⟩_L)
```

#### Exponential Map

Maps from tangent space to hyperboloid:

```
exp₀(v) = (cosh(||v||/√c), sinh(||v||/√c) * v/||v||)
```

### Contrastive Learning

The InfoNCE loss maximizes agreement between anchor-positive pairs while minimizing agreement with negatives:

```
L = -log(exp(sim(anchor, positive) / τ) / Σᵢ exp(sim(anchor, negativeᵢ) / τ))
```

In hyperbolic space, similarity is defined as negative distance:

```
sim(u, v) = -d(u, v)
```

---

## Design Decisions

### Why Hyperbolic Space?

1. **Hierarchical Structure**: Hyperbolic space naturally represents tree-like hierarchies
2. **Distance Properties**: Geodesic distances capture hierarchical relationships
3. **Capacity**: More capacity than Euclidean space for hierarchical data

### Why Lorentz Model?

1. **Differentiability**: Smooth operations for gradient-based learning
2. **Numerical Stability**: Well-conditioned distance computations
3. **Standard Form**: Widely used in machine learning literature

### Why Multi-Channel Encoding?

1. **Rich Semantics**: Different text fields capture different aspects
2. **Specialization**: Each channel can learn field-specific patterns
3. **Robustness**: Reduces reliance on any single text field

### Why MoE Fusion?

1. **Adaptive Combination**: Learns how to combine channels based on context
2. **Efficiency**: Top-k routing reduces computation
3. **Expressiveness**: Multiple experts capture diverse fusion patterns

### Why LoRA?

1. **Parameter Efficiency**: Reduces trainable parameters by ~90%
2. **Flexibility**: Can adapt any transformer architecture
3. **Memory Efficiency**: Enables larger batch sizes

### Why False-Negative Mitigation?

1. **Hierarchical Ambiguity**: Close codes in hierarchy may be sampled as negatives
2. **Curriculum Learning**: Gradually refines negative sampling as embeddings improve
3. **Better Representations**: Prevents model from incorrectly separating similar codes

---

## Component Dependencies

```
NAICSContrastiveModel
    ├─→ MultiChannelEncoder
    │   ├─→ 4x LoRA-adapted Transformers
    │   ├─→ MixtureOfExperts
    │   └─→ HyperbolicProjection
    ├─→ HyperbolicInfoNCELoss
    ├─→ HierarchyPreservationLoss (optional)
    ├─→ LambdaRankLoss (optional)
    └─→ Evaluation Components
        ├─→ EmbeddingEvaluator
        ├─→ EmbeddingStatistics
        └─→ HierarchyMetrics
```

---

## Configuration

Key hyperparameters (see `conf/config.yaml`):

- **Model**: `base_model_name`, `lora_r`, `lora_alpha`, `num_experts`, `top_k`
- **Hyperbolic**: `curvature`, `temperature`
- **Loss Weights**: `hierarchy_weight`, `rank_order_weight`, `radius_reg_weight`
- **Training**: `learning_rate`, `weight_decay`, `warmup_steps`
- **False Negatives**: `fn_curriculum_start_epoch`, `fn_cluster_every_n_epochs`, `fn_num_clusters`

---
