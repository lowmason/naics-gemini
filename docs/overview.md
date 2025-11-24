# NAICS Hyperbolic Embedding System Overview

This document provides a comprehensive overview of the NAICS Hyperbolic Embedding System, including its architecture, advanced features, and implementation details.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Advanced Features](#advanced-features)
4. [Data Flow](#data-flow)
5. [Training Pipeline](#training-pipeline)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Design Decisions](#design-decisions)
8. [Configuration](#configuration)

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

#### Key Features

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

#### Components

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

#### Projection Process

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

Contrastive learning is performed directly in hyperbolic space using **Decoupled Contrastive Learning (DCL)** with **Lorentzian geodesic distances**.

#### Decoupled Contrastive Learning (DCL) Loss

```python
HyperbolicInfoNCELoss(
    embedding_dim=embedding_dim,
    temperature=0.07,
    curvature=1.0
)
```

**Note**: Despite the class name, this loss function implements DCL rather than standard InfoNCE.

#### Distance Computation

Lorentzian distance between two points `u, v` on the hyperboloid:

```
d(u, v) = √c * arccosh(-⟨u, v⟩_L)
```

where the Lorentz inner product is:

```
⟨u, v⟩_L = u₁v₁ + ... + uₙvₙ - u₀v₀
```

#### Loss Function

Decoupled Contrastive Learning (DCL) loss with hyperbolic distances:

```
pos_sim = -d(anchor, positive) / τ
neg_sims = -d(anchor, negativeᵢ) / τ  for all i
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

where `τ` is the temperature parameter (default: `0.07`).

**Key Advantages of DCL:**

- **Decoupled gradients**: Positive and negative terms are computed separately, improving gradient flow
- **Numerical stability**: Uses `logsumexp` for stable computation of the negative term
- **Flexibility**: Can yield negative loss values (unlike InfoNCE), which is expected behavior

---

### 5. Additional Loss Components

#### Hierarchy Preservation Loss

Additional loss component that directly optimizes hierarchy preservation by matching embedding distances to tree distances.

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

#### Rank Order Preservation Loss (LambdaRank)

Global ranking optimization using LambdaRank to preserve rank order relationships.

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

#### Radius Regularization

Prevents hyperbolic embeddings from collapsing to the origin or expanding too far.

```
L_radius = radius_reg_weight * ||r - target_radius||²
```

where `r` is the hyperbolic radius (time coordinate `x₀`) and `target_radius` is a learned or fixed value.

**Default Weight**: `0.01`

---

### 6. Evaluation Metrics

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

## Advanced Features

### Hard Negative Mining

#### Overview

Hard negative mining selects the most challenging negatives for training by choosing negatives that are geometrically close to anchors in hyperbolic space. This provides a stronger learning signal than random negative sampling.

#### Implementation

```python
LorentzianHardNegativeMiner(
    curvature=1.0,
    safety_epsilon=1e-5
)
```

#### Process

1. For each anchor, compute Lorentzian distances to all candidate negatives
2. Select top-k negatives with smallest distances (hardest negatives)
3. Use these hard negatives in the contrastive loss

#### Benefits

- **Better Learning Signal**: Hard negatives provide more informative gradients
- **Faster Convergence**: Model learns to distinguish between similar codes more effectively
- **Improved Representations**: Embeddings develop finer-grained distinctions

#### Configuration

Enabled automatically when `enable_hard_negative_mining` is set in the curriculum scheduler.

---

### Router-Guided Negative Mining

#### Overview

Router-guided negative mining prevents "Expert Collapse" in the Mixture-of-Experts layer by selecting negatives that confuse the gating network. These negatives are identified by having similar expert probability distributions to anchors.

#### Implementation

```python
RouterGuidedNegativeMiner(
    metric='kl_divergence',  # or 'cosine_similarity'
    temperature=1.0
)
```

#### Process

1. Compute gate probabilities for anchors and negatives
2. Measure confusion using KL-divergence or cosine similarity
3. Select negatives with highest confusion (most similar gate distributions)
4. Mix router-hard negatives with embedding-hard negatives (default: 50/50)

#### Metrics

- **KL-Divergence**: Measures how different two probability distributions are
  - Lower KL-divergence = more confusion (similar distributions)
- **Cosine Similarity**: Measures the angle between two probability vectors
  - Higher cosine similarity = more confusion (similar directions)

#### Benefits

- **Prevents Expert Collapse**: Ensures all experts are utilized effectively
- **Diverse Negative Sampling**: Captures negatives that confuse the routing mechanism
- **Better MoE Training**: Improves expert specialization and load balancing

#### Configuration

Enabled automatically when `enable_router_guided_sampling` is set in the curriculum scheduler.

---

### Global Batch Sampling

#### Overview

Global batch sampling enables hard negative mining across all GPUs in distributed training. This is crucial for finding meaningful "Cousin" negatives that may not appear in small local batches (e.g., size 32).

#### Implementation

Automatically enabled when:
- Distributed training is active (`torch.distributed.is_initialized()`)
- Multiple GPUs available (`world_size > 1`)
- Hard negative mining or router-guided sampling is enabled

#### Process

1. **Gather Phase**: Collect negative embeddings from all GPUs using `torch.distributed.all_gather`
2. **Distance Computation**: Compute distances from local anchors to all global negatives
3. **Selection**: Select top-k hardest negatives from the global pool
4. **Gradient Flow**: Gradients flow back through the all_gather operation to all GPUs

#### Memory Management

**Example Configuration:**
- `batch_size=32`, `world_size=4`, `k_negatives=24`
- Global negatives: ~9MB per GPU
- Similarity matrix: ~393KB per batch

**Monitoring:**
- `train/global_batch/global_negatives_memory_mb`: Memory usage for global negatives
- `train/global_batch/similarity_matrix_memory_mb`: Memory usage for similarity matrix
- `train/global_batch/global_batch_size`: Effective global batch size
- `train/global_batch/global_k_negatives`: Number of negatives per anchor globally

#### Benefits

- **Larger Negative Pool**: Access to negatives from all GPUs, not just local batch
- **Better Hard Negatives**: More likely to find meaningful "Cousin" relationships
- **Improved Training**: Higher quality negative samples lead to better representations

#### Gradient Flow

The implementation uses `torch.distributed.all_gather` which preserves gradients:
- If input embeddings require gradients, gathered tensors also have gradients
- During backpropagation, gradients are scattered back to each rank
- All GPUs receive gradient updates for their embeddings

---

### Hyperbolic K-Means Clustering

#### Overview

Unlike standard Euclidean K-Means, the system uses Hyperbolic K-Means that operates directly in Lorentz space. This is more appropriate for hyperbolic embeddings and preserves geometric structure during clustering.

#### Implementation

```python
HyperbolicKMeans(
    n_clusters=500,
    curvature=1.0,
    max_iter=100,
    tol=1e-4
)
```

#### Process

1. Initialize cluster centroids in Lorentz space
2. Assign embeddings to nearest centroid using Lorentzian distances
3. Update centroids in hyperbolic space
4. Repeat until convergence

#### Benefits

- **Geometric Consistency**: Clusters respect hyperbolic geometry
- **Better False-Negative Detection**: More accurate cluster assignments
- **Preserves Structure**: Maintains hierarchical relationships during clustering

#### Usage

Used for false-negative mitigation:
1. Periodically cluster embeddings (default: every 5 epochs after epoch 10)
2. Identify negatives sharing cluster label with anchor
3. Mask these false negatives in the contrastive loss

---

### Norm-Adaptive Margins

#### Overview

Norm-adaptive margins adapt to the hyperbolic radius of anchors, providing more appropriate margins for different regions of hyperbolic space.

#### Formula

```
m(a) = m₀ * sech(||a||_L)
```

where:
- `m₀` is the base margin (default: 0.5)
- `||a||_L` is the Lorentz norm (hyperbolic radius) of anchor `a`
- `sech` is the hyperbolic secant function: `sech(x) = 1 / cosh(x)`

#### Behavior

- **Small Norm (Near Origin)**: Margin is close to base margin `m₀`
- **Large Norm (Far from Origin)**: Margin decreases as `sech(||a||_L)` approaches 0
- **Adaptive Difficulty**: Anchors near the leaf boundary (large norm) have smaller margins

#### Benefits

- **Adaptive Difficulty**: Margins adapt to the hyperbolic geometry
- **Geometric Awareness**: More appropriate margins for different regions of hyperbolic space
- **Better Training**: Prevents over-penalization of anchors far from origin

#### Configuration

Computed automatically when hard negative mining is enabled. Logged metrics:
- `train/curriculum/adaptive_margin_mean`: Mean adaptive margin
- `train/curriculum/adaptive_margin_min`: Minimum adaptive margin
- `train/curriculum/adaptive_margin_max`: Maximum adaptive margin

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
[Global Batch Sampling] (if distributed + hard negative mining enabled)
    ├─→ Gather negatives from all GPUs
    └─→ Create global negative pool
    ↓
[Hard Negative Mining] (if enabled)
    ├─→ Compute Lorentzian distances to all negatives
    ├─→ Select top-k hardest negatives
    └─→ (Optionally) Router-guided negative selection
    ↓
[Compute Distances]
    ├─→ Anchor-Positive distances
    └─→ Anchor-Negative distances (hard negatives)
    ↓
[Apply False-Negative Mask] (if available)
    ↓
[Decoupled Contrastive Learning (DCL) Loss]
    ↓
[Additional Losses]
    ├─→ Hierarchy Preservation Loss
    ├─→ LambdaRank Loss
    ├─→ Radius Regularization
    └─→ MoE Load Balancing Loss
    ↓
[Total Loss] → Backpropagation
    ↓
[Gradient Flow] → Updates embeddings on all GPUs (if distributed)
```

---

## Training Pipeline

### Structure-Aware Dynamic Curriculum

The system implements a **Structure-Aware Dynamic Curriculum** that progressively enables advanced training features based on training progress:

#### Curriculum Phases

**Phase 0 (Early Training)**: Basic contrastive learning
- Standard negative sampling
- No hard negative mining
- No false negative masking

**Phase 1 (Mid Training)**: Enhanced negative sampling
- Enable hard negative mining
- Enable false negative clustering
- Track negative sample type distribution

**Phase 2 (Advanced Training)**: Advanced techniques
- Enable router-guided sampling
- Mix embedding-hard and router-hard negatives
- Full curriculum features active

#### Features

- **Automatic Phase Transitions**: Phases activate based on epoch thresholds
- **Negative Sample Tracking**: Logs distribution of negative types (child/sibling/cousin/distant)
- **Smooth Progression**: Gradually introduces complexity as model improves

#### Configuration

Managed automatically by the `CurriculumScheduler` class. Phase transitions are based on:
- Current epoch
- Training progress
- Curriculum configuration

---

### Multi-Level Supervision

#### Overview

Multi-level supervision allows each anchor to have multiple positive examples at different hierarchy levels. This provides richer supervision signals and explicitly models relationships at different levels.

#### Implementation

- Batch is expanded so each positive level is a separate training item
- Loss naturally sums over all positive levels
- Provides gradient accumulation across hierarchy levels

#### Benefits

- **Rich Supervision**: Model learns from multiple positive relationships simultaneously
- **Hierarchy Awareness**: Explicitly models relationships at different levels
- **Better Representations**: Captures hierarchical structure more effectively

#### Usage

Enabled automatically when the dataset provides `positive_levels` in the batch.

---

### False-Negative Curriculum

After initial training (default: epoch 10), periodically:

1. Generate embeddings for all codes
2. Cluster embeddings using **Hyperbolic K-Means** in Lorentz space
3. Update pseudo-labels based on cluster assignments
4. Mask false negatives in subsequent training

**Hyperbolic K-Means Clustering:**

- Operates directly in Lorentz space using Lorentzian distances
- More appropriate for hyperbolic embeddings than Euclidean K-Means
- Preserves geometric structure during clustering

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

The system uses **Decoupled Contrastive Learning (DCL)** loss, which decouples the positive and negative terms for improved gradient flow:

```
pos_sim = -d(anchor, positive) / τ
neg_sims = [-d(anchor, negativeᵢ) / τ for all i]
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

In hyperbolic space, similarity is defined as negative distance:

```
sim(u, v) = -d(u, v)
```

**Key Differences from InfoNCE:**

- DCL computes `logsumexp(neg_sims)` directly rather than using the softmax normalization of InfoNCE
- The positive term is simply `-pos_sim` rather than being part of a log-softmax
- This decoupling provides better gradient flow and numerical stability
- DCL loss can be negative (unlike InfoNCE), which is expected behavior

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

## Configuration

Key hyperparameters (see `conf/config.yaml`):

- **Model**: `base_model_name`, `lora_r`, `lora_alpha`, `num_experts`, `top_k`
- **Hyperbolic**: `curvature`, `temperature`
- **Loss Weights**: `hierarchy_weight`, `rank_order_weight`, `radius_reg_weight`
- **Training**: `learning_rate`, `weight_decay`, `warmup_steps`, `use_warmup_cosine`
- **False Negatives**: `fn_curriculum_start_epoch`, `fn_cluster_every_n_epochs`, `fn_num_clusters`
- **Distributed Training**: `training.trainer.devices` (number of GPUs)

### Distributed Training

The system supports multi-GPU distributed training with automatic global batch sampling:

#### Setup

Configure the number of devices in `conf/config.yaml`:

```yaml
training:
  trainer:
    devices: 4  # Number of GPUs
    accelerator: 'gpu'
```

#### Global Batch Sampling

When distributed training is enabled with hard negative mining:

- **Automatic Activation**: Global batch sampling activates automatically
- **Memory Efficient**: Monitors and logs VRAM usage
- **Gradient Flow**: Gradients flow back through all_gather to all GPUs
- **Better Negatives**: Access to negatives from all GPUs, not just local batch

#### Monitoring

TensorBoard logs include:

- `train/global_batch/global_negatives_memory_mb`: Memory usage for global negatives
- `train/global_batch/similarity_matrix_memory_mb`: Memory usage for similarity matrix
- `train/global_batch/global_batch_size`: Effective global batch size
- `train/global_batch/global_k_negatives`: Number of negatives per anchor globally

---

## Component Dependencies

```
NAICSContrastiveModel
    ├─→ MultiChannelEncoder
    │   ├─→ 4x LoRA-adapted Transformers
    │   ├─→ MixtureOfExperts
    │   └─→ HyperbolicProjection
    ├─→ HyperbolicInfoNCELoss (implements DCL)
    ├─→ HierarchyPreservationLoss (optional)
    ├─→ LambdaRankLoss (optional)
    └─→ Evaluation Components
        ├─→ EmbeddingEvaluator
        ├─→ EmbeddingStatistics
        └─→ HierarchyMetrics
```

---

## Summary

The NAICS Hyperbolic Embedding System integrates multiple advanced features that work together to improve training:

1. **Hard Negative Mining** provides challenging negatives for better learning
2. **Router-Guided Sampling** prevents expert collapse in MoE
3. **Global Batch Sampling** enables access to negatives from all GPUs
4. **Structure-Aware Curriculum** gradually introduces complexity
5. **Multi-Level Supervision** provides richer training signals
6. **Hyperbolic K-Means** improves false-negative detection
7. **Norm-Adaptive Margins** adapt to hyperbolic geometry

All features are automatically enabled based on training configuration and progress, requiring no manual intervention.

