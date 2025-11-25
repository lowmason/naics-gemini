# NAICS Hyperbolic Embedding System

**Unified Framework for Hierarchical Representation Learning**

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Multi-Channel Text Encoding](#3-multi-channel-text-encoding)
4. [Mixture-of-Experts Fusion](#4-mixture-of-experts-fusion)
5. [Hyperbolic Geometry & Lorentz Model](#5-hyperbolic-geometry--lorentz-model)
6. [Contrastive Learning Framework](#6-contrastive-learning-framework)
7. [Sampling Strategies](#7-sampling-strategies)
8. [Curriculum Learning (SADC)](#8-structure-aware-dynamic-curriculum-sadc)
9. [False Negative Mitigation](#9-false-negative-mitigation)
10. [Additional Loss Components](#10-additional-loss-components)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Distributed Training](#12-distributed-training)
13. [Implementation Reference](#13-implementation-reference)

---

## 1. Overview

The NAICS Hyperbolic Embedding System is a unified framework for learning hierarchical representations of the North American Industry Classification System (NAICS) taxonomy. The system addresses a fundamental challenge in representation learning: embedding tree-structured categorical data into a continuous vector space while preserving hierarchical relationships.

Unlike standard classification approaches that treat categories as equidistant entities, this system recognizes that NAICS codes exist within a rich taxonomic structure spanning from broad Sectors (2-digit) to precise National Industries (6-digit). The semantic distance between sibling codes like 541511 (Custom Computer Programming) and 541512 (Computer Systems Design) is fundamentally different from the distance to 111110 (Soybean Farming).

### Key Architectural Decisions

**1. Hyperbolic Geometry (Lorentz Model):** Euclidean space is geometrically incompatible with tree structures—tree nodes grow exponentially with depth while Euclidean volume grows only polynomially. Hyperbolic space, with its exponential volume growth, provides a natural, low-distortion embedding environment for hierarchies. The Lorentz model is chosen over the Poincaré ball for its superior numerical stability.

**2. Mixture-of-Experts Fusion:** Each NAICS code has four text channels (title, description, examples, excluded) with heterogeneous informativeness. MoE with Top-2 gating enables learning multiple specialized fusion strategies, allowing different experts to handle different types of codes.

**3. Curriculum-Based Training:** A three-phase Structure-Aware Dynamic Curriculum (SADC) progressively introduces complexity: structural initialization → geometric refinement → false negative mitigation.

**4. Decoupled Contrastive Learning:** DCL provides better gradient flow and numerical stability compared to standard InfoNCE, with the loss computed as:

```
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

---

## 2. System Architecture Overview

The system consists of four sequential stages, each designed to preserve or enhance the hierarchical geometry of NAICS codes:

| Stage | Component | Output |
|-------|-----------|--------|
| 1 | Multi-Channel Text Encoding (4 LoRA-adapted transformers) | E_title, E_desc, E_examples, E_excluded (4 × embedding_dim) |
| 2 | Mixture-of-Experts Fusion (Top-2 gating, 4 experts) | E_fused (embedding_dim) |
| 3 | Hyperbolic Projection (Lorentz exponential map) | E_hyp (embedding_dim + 1) |
| 4 | Contrastive Learning (DCL + auxiliary losses) | Trained embeddings on Lorentz hyperboloid |

### Data Flow Diagram

```
NAICS Code (4 text channels)
        ↓
[Multi-Channel Encoder]
    ├─→ Title Encoder (LoRA) → E_title
    ├─→ Description Encoder (LoRA) → E_desc
    ├─→ Examples Encoder (LoRA) → E_examples
    └─→ Excluded Encoder (LoRA) → E_excluded
        ↓
[Concatenate] → (embedding_dim × 4)
        ↓
[MoE Fusion] → E_fused (embedding_dim)
        ↓
[Hyperbolic Projection] → E_hyp (embedding_dim + 1)
        ↓
[Lorentz Hyperboloid] → Final Embedding
```

---

## 3. Multi-Channel Text Encoding

Each NAICS code is characterized by four distinct text fields, each providing complementary information about the industry classification:

| Channel | Content | Purpose |
|---------|---------|---------|
| Title | Short code name (e.g., "Software Publishers") | Concise category identification |
| Description | Detailed explanation of what the code encompasses | Rich semantic content |
| Examples | Representative businesses in this category | Concrete instantiations |
| Excluded | Codes explicitly NOT in this category | Disambiguation and boundaries |

### LoRA Adaptation

Each channel uses a separate LoRA-adapted transformer encoder based on sentence-transformers. LoRA (Low-Rank Adaptation) reduces trainable parameters while maintaining expressiveness:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| base_model | all-mpnet-base-v2 | Pre-trained sentence transformer |
| lora_r | 8 | LoRA rank (lower = fewer parameters) |
| lora_alpha | 16 | LoRA scaling factor |
| lora_dropout | 0.1 | Dropout rate for regularization |
| target_modules | all-linear | Universal targeting for any transformer |

Gradient checkpointing is enabled by default to reduce memory usage during backpropagation, which is critical for large batch sizes or limited GPU memory.

---

## 4. Mixture-of-Experts Fusion

The relative importance of text channels varies across NAICS codes. For some codes, the title and description suffice; for others, the examples are most illustrative; for nuanced codes, the excluded field is critical for disambiguation. A static fusion strategy cannot adapt to this heterogeneity.

### Why MoE Over Alternatives

Three fusion strategies were evaluated: learned weighted average (static), gated attention (continuous dynamic control), and Mixture-of-Experts (discrete dynamic selection). MoE provides the most powerful paradigm because it offers coarse-grained selection between multiple specialized processing paths, not just dynamic weighting.

The MoE framework allows the model to effectively perform a learned architectural search, discovering experts optimized for different input types. One expert might specialize in ambiguity resolution (up-weighting the excluded channel), while another becomes a general classification expert focused on title and description.

### Architecture

| Component | Configuration | Function |
|-----------|---------------|----------|
| Input | embedding_dim × 4 | Concatenated channel embeddings |
| Gating Network | Linear(input → num_experts) | Computes expert selection scores |
| Top-K Selection | k = 2 | Selects 2 most relevant experts |
| Expert Networks | 4 × 2-layer MLP | Linear→ReLU→Dropout→Linear |
| Hidden Dim | 1024 | Expert network hidden dimension |
| Output Projection | Linear(input → embedding_dim) | Projects back to embedding space |

### Load Balancing Loss

Without correction, gating networks favor a small subset of "winning" experts, causing mode collapse. An auxiliary load balancing loss ensures even utilization:

```
L_aux = α · N · Σ(f_i · P_i)
```

Where N is the number of experts, α = 0.01 (default coefficient), f_i is the fraction of tokens routed to expert i, and P_i is the average gating probability for expert i.

### Global-Batch vs. Micro-Batch Statistics

A critical implementation detail: the auxiliary loss must be calculated on **global-batch** statistics, not micro-batch. Micro-batch balancing forces the router to balance within each sequence, hindering domain specialization. Global-batch balancing allows the router to send all "manufacturing" codes to Expert 1 and all "healthcare" codes to Expert 2, as long as total utilization remains balanced across the entire diverse batch.

This requires synchronizing expert utilization counts (f_i) and router probabilities (P_i) across all distributed workers via AllReduce before computing the loss.

---

## 5. Hyperbolic Geometry & Lorentz Model

### The Geometric Mismatch Problem

Attempting to embed hierarchical data into Euclidean space faces a fundamental geometric incompatibility: the number of nodes in a tree grows exponentially with depth (~ b^L for branching factor b and depth L), while the volume of a Euclidean ball grows only polynomially with radius (~ r^d). This disparity inevitably leads to distortion.

Hyperbolic geometry provides a principled solution. Hyperbolic spaces have constant negative curvature, causing volume to grow exponentially with radius (~ e^r). This makes hyperbolic space a natural, parsimonious, low-distortion environment for embedding hierarchies.

### The Lorentz Model

Two common models of hyperbolic space are the Poincaré Ball and the Lorentz (Hyperboloid) Model. The Lorentz model is chosen for its superior numerical stability—the Poincaré model suffers from "the NaN problem" as embeddings approach the boundary.

The Lorentz model represents points as (x₀, x₁, ..., xₙ) on a hyperboloid satisfying:

```
-x₀² + x₁² + ... + xₙ² = -1/c
```

Where x₀ is the time coordinate (hyperbolic radius), x₁...xₙ are spatial coordinates, and c is the curvature parameter (default: c = 1.0).

### Key Operations

**Lorentz Inner Product:**

```
⟨u, v⟩_L = u₁v₁ + ... + uₙvₙ - u₀v₀
```

**Lorentzian Distance (Geodesic):**

```
d(u, v) = √c · arccosh(-⟨u, v⟩_L)
```

**Exponential Map (Tangent → Hyperboloid):**

```
x₀ = cosh(||v|| / √c)
x_rest = (sinh(||v|| / √c) · v) / ||v||
```

### Hyperbolic Projection Implementation

The fused Euclidean embedding is projected onto the hyperboloid via a linear projection followed by the exponential map at the origin. The projection adds the time coordinate dimension (embedding_dim → embedding_dim + 1) and ensures points satisfy the Lorentz constraint through numerically stable clamping.

---

## 6. Contrastive Learning Framework

### Decoupled Contrastive Learning (DCL)

The system uses Decoupled Contrastive Learning rather than standard InfoNCE. DCL decouples the positive and negative terms for improved gradient flow and numerical stability:

```
pos_sim = -d(anchor, positive) / τ
neg_sims = [-d(anchor, negative_i) / τ for all i]
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

Where τ is the temperature parameter (default: 0.07). In hyperbolic space, similarity is defined as negative Lorentzian distance.

### Key Differences from InfoNCE

| Aspect | InfoNCE | DCL |
|--------|---------|-----|
| Formulation | log(exp(pos) / Σexp(all)) | -pos + logsumexp(neg) |
| Coupling | Positive in denominator | Decoupled terms |
| Loss Range | Always ≥ 0 | Can be negative |
| Gradient Flow | Coupled gradients | Independent gradients |

### Gradient Analysis

The gradient magnitude with respect to a negative sample n is proportional to its probability weight in the softmax distribution:

```
w_in = exp(-d_L(z_i, z_n) / τ) / Z_i
```

This mathematical structure dictates the informational value of negatives. **Easy negatives** (d >> d_pos) contribute near-zero gradient. **Hard negatives** (d ≈ d_pos) provide strong learning signal. **Collapsing negatives** (d < d_pos) represent current errors and yield maximal gradients.

### False Negative Masking

When false negatives are detected (samples from different but semantically related classes), they are masked from the loss computation using the elimination strategy—setting their similarities to -∞ rather than re-categorizing them as positives. This is more robust to noise in pseudo-labels.

```python
neg_similarities = neg_similarities.masked_fill(false_negative_mask, -inf)
```

---

## 7. Sampling Strategies

The sampling strategy fundamentally governs learning dynamics. In dense hierarchical taxonomies like NAICS, the definition of "negative" is fluid and context-dependent.

### The Gradient-Semantic Trade-off

Standard contrastive learning treats all negatives equally. However, a model initialized with random weights will immediately separate "Farming" from "Programming" based on coarse lexical features. Triplets with distant negatives quickly satisfy the margin condition, driving loss to zero and extinguishing gradient signal.

To learn fine-grained features distinguishing "Custom Programming" from "Systems Design", sampling must mine negatives from the local neighborhood—"cousins" and "siblings" of the hierarchy. Yet pushing semantically proximal nodes apart risks shattering cluster structure.

### Negative Type Taxonomy

| Type | Tree Distance | Gradient | Risk | Recommendation |
|------|---------------|----------|------|----------------|
| Siblings | d = 2 | Very High | False Negative | Mask in Phase 1 |
| Cousins | d = 4 | High | Low | Optimal negatives |
| 2nd Cousins | d = 6 | Medium | Very Low | Good negatives |
| Distant | d ≥ 8 | Near Zero | None | Low utility |

### Hard Negative Mining

Embedding-based hard negative mining dynamically selects negatives that are currently close to the anchor in hyperbolic space. The LorentzianHardNegativeMiner computes distances to all candidate negatives and selects the top-k with smallest distances.

This adapts to the model's current state, targeting exact boundaries where the model is confused. However, it risks the "False Negative Trap" in hierarchical data—embeddings closest to an anchor are likely siblings or cousins, which are semantically similar.

### Router-Guided Sampling

Router-guided sampling selects negatives that maximize confusion in the MoE gating network. If the router sends anchor and negative to the same experts with similar confidence, they are "computationally indistinguishable." Using these as contrastive negatives forces experts to become more discriminative and combats mode collapse.

### Global Batch Sampling

A local micro-batch (e.g., size 32 per GPU) is statistically unlikely to contain "Cousin" negatives (distance-4). Cross-device negative sampling gathers embeddings from all GPUs to create a larger candidate pool, enabling selection of meaningful hard negatives.

---

## 8. Structure-Aware Dynamic Curriculum (SADC)

The optimal sampling strategy is not a single static configuration but a dynamic, structure-aware process that evolves over training. The SADC implements three phases:

### Phase 1: Structural Initialization (0-30%)

**Objective:** Establish global topology and local clustering based on the explicit NAICS tree.

**Strategy:** Tree-Distance Weighted Sampling with Sibling Masking

```
P_S1(n|a) ∝ 1/d_tree(a,n)^α · 𝟙(d_tree(a,n) > 2)
```

Inverse distance weighting (α ≈ 1.5) biases selection toward "Cousins" (d=4). Siblings (d=2) are explicitly masked—treating siblings as negatives early in training is dangerous because the model lacks feature maturity to distinguish them subtly.

**Curriculum Flags:** `use_tree_distance=True`, `mask_siblings=True`

### Phase 2: Geometric Refinement (30-70%)

**Objective:** Refine decision boundaries using the learned metric space.

**Strategy:** Annealed Hard Negative Mining in Lorentz Space

As the embedding space matures, transition from symbolic tree priors to learned semantics. Sample a candidate pool, then select top-k negatives minimizing Lorentzian distance. Router-guided sampling is also enabled to force expert specialization.

**Curriculum Flags:** `enable_hard_negative_mining=True`, `enable_router_guided_sampling=True`

### Phase 3: False Negative Mitigation (70-100%)

**Objective:** Clean embedding space of artifacts; resolve semantic ambiguities.

**Strategy:** Clustering-Based False Negative Elimination (FNE)

Periodically freeze the encoder and perform Hyperbolic K-Means clustering. Assign cluster IDs as pseudo-labels. When sampling negatives, if Cluster(anchor) == Cluster(negative), eliminate that negative from the loss. This accepts that some distinct codes are semantically identical and stops fighting the data.

**Curriculum Flags:** `enable_clustering=True`

### Phase Transition Summary

| Phase | Epochs | Key Features | Goal |
|-------|--------|--------------|------|
| 1 | 0-30% | Tree-distance weighting, sibling masking | Build skeleton |
| 2 | 30-70% | Hard negative mining, router-guided sampling | Refine shape |
| 3 | 70-100% | Clustering-based FNE | Clean artifacts |

---

## 9. False Negative Mitigation

### The False Negative Problem

In contrastive learning, a "false negative" is a sample treated as negative despite being semantically similar to the anchor. This problem is acute for NAICS: given anchor 541511 (Custom Computer Programming), sibling 541512 (Computer Systems Design) is semantically very close. Standard contrastive loss would incorrectly apply repulsive force, damaging hierarchical structure.

The detrimental effect is pronounced in large-scale datasets with high semantic concept density—a perfect description of NAICS. Consequences include discarding valuable shared semantic information and slowed convergence.

### Why Curriculum-Based Detection

Attempting false negative detection too early is counterproductive. In initial training, the embedding space is largely random—any "semantic neighbors" identified via clustering would be spurious. The detection mechanism should activate only after the embedding space has stabilized (typically 70% of training).

This creates a self-correction loop: the model first learns coarse representations, then uses that emergent structure to identify and correct inconsistencies in its own training objective, then refines representations based on this more accurate objective.

### Detection via Hyperbolic K-Means

Unlike standard Euclidean K-Means, the system uses Hyperbolic K-Means operating directly in Lorentz space. This is more appropriate for hyperbolic embeddings and preserves geometric structure during clustering.

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_clusters | 500 | Number of semantic clusters |
| curvature | 1.0 | Lorentz model curvature |
| max_iter | 100 | Maximum K-Means iterations |
| tol | 1e-4 | Convergence tolerance |
| update_frequency | 5 epochs | Re-clustering interval in Phase 3 |

### Elimination vs. Attraction Strategy

Two mitigation strategies exist after identifying false negatives. **Elimination** removes false negatives from the denominator—the model ignores them. **Attraction** re-categorizes them as positives in the numerator—the model pulls them closer.

Research indicates attraction is less tolerant to noise in pseudo-labels. Since clustering-based detection inevitably produces some noise, **elimination is the recommended and implemented strategy**.

---

## 10. Additional Loss Components

Beyond the primary DCL contrastive loss, the system includes several auxiliary losses to enforce specific geometric and structural properties:

### Hierarchy Preservation Loss

Directly optimizes embedding distances to match ground-truth tree distances:

```
L_hierarchy = weight · MSE(d_embedding, d_tree)
```

For each pair of codes in the batch, the loss penalizes deviations between Lorentzian geodesic distance and NAICS tree distance. Default weight: 0.325.

### LambdaRank Loss (Rank Order Preservation)

Global ranking optimization using LambdaRank to preserve rank-order relationships.

Unlike pairwise losses, LambdaRank optimizes NDCG@k (Normalized Discounted Cumulative Gain), weighting pairs by their impact on ranking position. This provides position-aware optimization considering all pairs, not just anchor-positive-negative triplets. Default weight: 0.275.

### Radius Regularization

Prevents hyperbolic embeddings from collapsing to the origin or expanding too far:

```
L_radius = weight · ||r - target_radius||²
```

Where r is the hyperbolic radius (time coordinate x₀). Default weight: 0.01.

### MoE Load Balancing Loss

As described in Section 4, ensures even expert utilization:

```
L_aux = α · N · Σ(f_i · P_i)
```

Default coefficient α = 0.01.

### Total Loss

```
L_total = L_DCL + L_hierarchy + L_lambdarank + L_radius + L_load_balancing
```

| Loss Component | Default Weight | Purpose |
|----------------|----------------|---------|
| DCL Contrastive | 1.0 (implicit) | Primary representation learning |
| Hierarchy Preservation | 0.325 | Tree structure alignment |
| LambdaRank | 0.275 | Rank-order preservation |
| Radius Regularization | 0.01 | Embedding stability |
| Load Balancing | 0.01 | Expert utilization balance |

---

## 11. Evaluation Metrics

The system computes comprehensive evaluation metrics during training to monitor hierarchy preservation, embedding quality, and potential failure modes.

### Hierarchy Preservation Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| Cophenetic Correlation | Correlation between embedding and tree distances | → 1.0 |
| Spearman Correlation | Rank-order correlation of distance pairs | → 1.0 |
| NDCG@5 | Ranking quality (top 5 neighbors) | → 1.0 |
| NDCG@10 | Ranking quality (top 10 neighbors) | → 1.0 |
| NDCG@20 | Ranking quality (top 20 neighbors) | → 1.0 |
| Mean Distortion | Average distance distortion from tree | → 0.0 |

### Hyperbolic Geometry Metrics

| Metric | Description | Notes |
|--------|-------------|-------|
| Lorentz Norm Mean | Average ⟨x,x⟩_L across embeddings | Should be ≈ -1/c |
| Lorentz Norm Violations | Points violating hyperboloid constraint | Should be 0 |
| Hyperbolic Radius Mean | Average x₀ (time coordinate) | Indicates hierarchy depth |
| Hyperbolic Radius Std | Standard deviation of radii | Indicates spread |

### Collapse Detection

The system monitors for embedding collapse, where all embeddings converge to a single point or small region, indicating training failure:

| Metric | Description | Warning Threshold |
|--------|-------------|-------------------|
| Norm CV | Coefficient of variation of norms | < 0.1 indicates collapse |
| Distance CV | Coefficient of variation of pairwise distances | < 0.1 indicates collapse |
| Variance Collapse | Boolean flag for detected collapse | True = problem |

---

## 12. Distributed Training

### Multi-GPU Support

The system supports distributed training with automatic global batch sampling. Key features:

**Global Negative Gathering:** When hard negative mining or router-guided sampling is enabled, negative embeddings are gathered from all GPUs using `torch.distributed.all_gather`. This creates a much larger candidate pool for hard negative selection.

**Gradient Flow:** The implementation preserves gradients through all_gather operations. During backpropagation, gradients are scattered back to each rank, ensuring all GPUs receive gradient updates for their embeddings.

**Global-Batch Load Balancing:** Expert utilization statistics are synchronized across all workers via AllReduce before computing the auxiliary loss, enabling true domain specialization.

### Memory Management

The system monitors and logs VRAM usage for distributed operations:

| Metric | Example (batch=32, world=4, k=24) |
|--------|-----------------------------------|
| train/global_batch/global_negatives_memory_mb | ~9 MB per GPU |
| train/global_batch/similarity_matrix_memory_mb | ~393 KB per batch |
| train/global_batch/global_batch_size | 128 (32 × 4) |
| train/global_batch/global_k_negatives | 96 (24 × 4) |

---

## 13. Implementation Reference

### Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `NAICSContrastiveModel` | `text_model/naics_model.py` | Main Lightning module |
| `MultiChannelEncoder` | `text_model/encoder.py` | 4-channel text encoding |
| `MixtureOfExperts` | `text_model/moe.py` | MoE fusion layer |
| `HyperbolicProjection` | `text_model/hyperbolic.py` | Lorentz projection |
| `LorentzDistance` | `text_model/hyperbolic.py` | Geodesic distance |
| `HyperbolicInfoNCELoss` | `text_model/loss.py` | DCL implementation |
| `HierarchyPreservationLoss` | `text_model/loss.py` | Tree alignment loss |
| `LambdaRankLoss` | `text_model/loss.py` | Ranking loss |
| `CurriculumScheduler` | `text_model/curriculum.py` | SADC phase management |
| `HyperbolicKMeans` | `text_model/hyperbolic_clustering.py` | Lorentz clustering |
| `LorentzianHardNegativeMiner` | `text_model/hard_negative_mining.py` | HNM in hyperbolic space |

### Default Hyperparameters

| Category | Parameter | Default |
|----------|-----------|---------|
| Model | base_model_name | all-mpnet-base-v2 |
| LoRA | r / alpha / dropout | 8 / 16 / 0.1 |
| MoE | num_experts / top_k / hidden_dim | 4 / 2 / 1024 |
| Loss | temperature / curvature | 0.07 / 1.0 |
| Loss Weights | hierarchy / rank_order / radius_reg | 0.325 / 0.275 / 0.01 |
| MoE | load_balancing_coef | 0.01 |
| Training | learning_rate / weight_decay | 2e-4 / 0.01 |
| Training | warmup_steps | 500 |
| Curriculum | phase1_end / phase2_end | 0.3 / 0.7 |
| Clustering | n_clusters / update_freq | 500 / 5 epochs |

### CLI Commands

```bash
# Data preprocessing
uv run naics-embedder data all

# Training
uv run naics-embedder train
```

---

## Appendix A: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| ⟨u, v⟩_L | Lorentz inner product |
| d_L(u, v) | Lorentzian geodesic distance |
| d_tree(a, n) | Tree distance (shortest path in NAICS taxonomy) |
| τ | Temperature parameter |
| c | Curvature parameter |
| x₀ | Time coordinate (hyperbolic radius) |
| f_i | Fraction of tokens routed to expert i |
| P_i | Average gating probability for expert i |
| α | Load balancing coefficient |

---

## Appendix B: Literature References

**Hyperbolic Deep Learning & Graph Neural Networks ## Hyperbolic Geometry**

- Chami et al. (2019). [Hyperbolic Graph Convolutional Neural Networks.](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf)
- Liu et al. (2019). [Hyperbolic Graph Neural Networks.](https://proceedings.neurips.cc/paper_files/paper/2019/file/103303dd56a731e377d01f6a37badae3-Paper.pdf)
- Nickel & Kiela (2017). [Poincaré Embeddings for Learning Hierarchical Representations.](https://papers.nips.cc/paper_files/paper/2017/file/59dfa2df42d9e3d41f5b02bfc32229dd-Paper.pdf)
- Nickel & Kiela (2018). [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry.](https://proceedings.mlr.press/v80/nickel18a/nickel18a.pdf)
- Ganea et al. (2018). [Hyperbolic Neural Networks.](https://proceedings.neurips.cc/paper_files/paper/2018/file/dbab2adc8f9d078009ee3fa810bea142-Paper.pdf)
- Dai et al. (2021). [A Hyperbolic-to-Hyperbolic Graph Convolutional Network.](https://arxiv.org/pdf/2104.06942)

**Contrastive Learning**

- Yeh et al. (2022). [Decoupled Contrastive Learning.](https://arxiv.org/pdf/2110.06848)
- Chen et al. (2020). [A Simple Framework for Contrastive Learning of Visual Representations.](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf)
- Khosla et al. (2020). [Supervised Contrastive Learning.](https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)
- Ge et al. (2023). [Hyperbolic Contrastive Learning for Visual Representations beyond Objects.](https://arxiv.org/pdf/2212.00653)
- Robinson et al. (2021). [Contrastive Learning with Hard Negative Samples.](https://arxiv.org/abs/2010.04592)
- Zhang et al. (2022). [Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework.](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Use_All_the_Labels_A_Hierarchical_Multi-Label_Contrastive_Learning_Framework_CVPR_2022_paper.pdf)
- Ahrabian et al. (2020). [Structure Aware Negative Sampling in Knowledge Graphs.](https://www.researchgate.net/publication/344373367_Structure_Aware_Negative_Sampling_in_Knowledge_Graphs)
- Alon et al. (2024). [Optimal Sample Complexity of Contrastive Learning.](https://openreview.net/forum?id=NU9AYHJvYe)

**Mixture-of-Experts**

- Shazeer et al. (2017). [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.](https://openreview.net/pdf?id=B1ckMDqlg)
- Fedus et al. (2022). [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf)
- Jacobs et al. (1991). [Adaptive Mixtures of Local Experts.](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf)

**NAICS & Industry Classification**

- Whitehead & Dumbacher (2024). [Ensemble Modeling Techniques for NAICS Classification in the Economic Census.](https://www.census.gov/library/working-papers/2024/econ/ensemble-modeling-techniques-for-naics-classification-in-the-economic-census.html)
- Vidali et al. (2024). [Unlocking NACE Classification Embeddings with OpenAI for Enhanced Analysis.](https://arxiv.org/abs/2409.11524)

**Text Encoding & Parameter Efficiency**

- Vaswani et al. (2017). [Attention Is All You Need.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Hu et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models.](https://arxiv.org/pdf/2106.09685)
- Reimers & Gurevych (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.](https://aclanthology.org/D19-1410.pdf)

---
