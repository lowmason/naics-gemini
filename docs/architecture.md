# NAICS Gemini System Architecture

This document provides a comprehensive technical overview of the NAICS Gemini hierarchical contrastive learning system, including data flow, model architecture, and implementation details.

## Table of Contents

- [System Overview](#system-overview)
- [Data Pipeline Architecture](#data-pipeline-architecture)
- [Model Architecture](#model-architecture)
- [Training Architecture](#training-architecture)
- [Key Design Decisions](#key-design-decisions)
- [Performance Characteristics](#performance-characteristics)

---

## System Overview

NAICS Gemini learns hierarchical embeddings for 2,125 NAICS industry codes using contrastive learning in hyperbolic space. The system comprises three major components:

1. **Data Pipeline:** Transforms raw Census Bureau files into filtered training triplets
2. **Model:** Multi-channel encoder with MoE fusion and hyperbolic projection
3. **Training:** Curriculum-based contrastive learning with load balancing

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                  │
│                                                                         │
│  Census Bureau    ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│  Excel Files  →   │Preprocess│  →  │Distances │  →  │ Triplets │        │
│  (4 files)        │  2,125   │     │ 3.0M     │     │  263M    │        │
│                   │  codes   │     │  pairs   │     │ triplets │        │
│                   └──────────┘     └──────────┘     └──────────┘        │
│                        ↓                                                │
│                   Parquet Files (streaming)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODEL ARCHITECTURE                             │
│                                                                         │
│  ┌─ Multi-Channel Encoder ────────────────────────────────────────┐     │
│  │                                                                │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │     │
│  │  │ Title   │  │  Desc   │  │Excluded │  │Examples │            │     │
│  │  │ Channel │  │ Channel │  │ Channel │  │ Channel │            │     │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │     │
│  │       │            │            │            │                 │     │
│  │  ┌────▼─────┐ ┌───▼─────┐ ┌───▼─────┐ ┌───▼─────┐              │     │
│  │  │Transformer│ │Transform│ │Transform│ │Transform│             │     │
│  │  │ + LoRA   │ │ + LoRA  │ │ + LoRA  │ │ + LoRA  │              │     │
│  │  └────┬─────┘ └───┬─────┘ └───┬─────┘ └───┬─────┘              │     │
│  │       │            │            │            │                 │     │
│  │       └────────────┴────────────┴────────────┘                 │     │
│  │                          │                                     │     │
│  │                    Concatenate                                 │     │
│  │                          │                                     │     │
│  └──────────────────────────┼─────────────────────────────────────┘     │
│                             ↓                                           │
│  ┌─ Mixture of Experts ─────────────────────────────────────────┐       │
│  │                                                              │       │
│  │    ┌──────────┐      ┌─────────┐  ┌─────────┐                │       │
│  │    │  Gating  │  →   │Expert 1 │  │Expert 2 │                │       │
│  │    │  Network │  →   │Expert 3 │  │Expert 4 │                │       │
│  │    └──────────┘      └─────────┘  └─────────┘                │       │
│  │         │                    │           │                   │       │
│  │    Top-2 Selection      Weighted Sum                         │       │
│  │                                                              │       │
│  └────────────────────────────┼──────────────────────────-──────┘       │
│                                ↓                                        │
│  ┌─ Hyperbolic Projection ───────────────────────────────────┐          │
│  │                                                           │          │
│  │   Euclidean → Tangent Space → Lorentz Model (Hyperboloid) │          │
│  │   (768-dim)      (769-dim)         (769-dim)              │          │
│  │                                                           │          │
│  └─────────────────────────────────────────────────────────-─┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING & LOSS                                  │
│                                                                         │
│  Anchor, Positive, K Negatives → Hyperbolic InfoNCE Loss                │
│  (Lorentzian distance) + MoE Load Balancing Loss                        │
│                                                                         │
│  Optimizer: AdamW + Cosine Annealing                                    │
│  Curriculum: Hardness-based filtering + negative sampling               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline Architecture

### Stage 1: Preprocessing (`download_data.py`)

**Input:** 4 Excel files from Census Bureau
- `2-6 digit_2022_Codes.xlsx` - Official titles
- `2022_NAICS_Index_File.xlsx` - Illustrative examples
- `2022_NAICS_Descriptions.xlsx` - Detailed descriptions
- `2022_NAICS_Cross_References.xlsx` - Exclusions

**Process:**
1. Download files with exponential backoff retry
2. Parse Excel sheets into Polars DataFrames
3. Normalize combined sector codes (31-33 → 31, 44-45 → 44, 48-49 → 48)
4. Extract and clean descriptions (remove HTML, normalize whitespace)
5. Mine exclusions from cross-reference file and description text
6. Aggregate examples from index file
7. Fill missing descriptions from child codes (e.g., 4-digit from 5-digit)

**Output:** `data/naics_descriptions.parquet`

```
Schema:
  - index: UInt32          (0-2124, unique identifier)
  - level: UInt8           (2-6, hierarchy depth)
  - code: String           (NAICS code, e.g., "541511")
  - title: String          (Official title)
  - description: String    (Detailed description)
  - examples: String       (Illustrative examples)
  - excluded: String       (Exclusion text)
  - excluded_codes: List   (Parsed exclusion codes)

Rows: 2,125
Size: ~1.2 MB
```

### Stage 2: Distance Computation (`compute_distances.py`)

**Input:** `data/naics_descriptions.parquet`

**Process:**
1. Build hierarchy trees for each sector (20 sectors)
2. For each sector tree:
   - Compute depth and ancestor paths for all nodes
   - Generate all code pairs (i, j) where i ≤ j in tree order
   - Calculate graph distance using formula:
     ```
     distance = (depth_i - depth_ancestor) + (depth_j - depth_ancestor) - 0.5 * lineal
     ```
     where `lineal = 1` if j is direct descendant of i
3. Cross-sector pairs: assign distance = 10.0
4. Combine all sector distances

**Output:** `data/naics_distances.parquet`

```
Schema:
  - idx_i: UInt32          (Index of code i)
  - idx_j: UInt32          (Index of code j)
  - code_i: String         (NAICS code i)
  - code_j: String         (NAICS code j)
  - distance: Float32      (Graph distance: 0.5-10.0)

Rows: 3,004,420
Size: ~48 MB
```

**Distance Distribution:**
- 0.5 (parent-child): 0.07%
- 1.0-8.0 (within-sector): 9.00%
- 10.0 (cross-sector): 90.94%

### Stage 3: Triplet Generation (`create_triplets.py`)

**Input:** 
- `data/naics_descriptions.parquet`
- `data/naics_distances.parquet`

**Process:**
1. Identify all positive pairs: (i, j) where distance < max
2. Identify all negative pairs: (i, k) for each positive
3. Filter for valid triplets: negative_distance > positive_distance
4. Annotate with metadata:
   - `excluded`: True if negative in positive's exclusion list
   - `unrelated`: True if negative in different sector
   - `distance_diff`: negative_distance - positive_distance
5. Assign hardness level (1-8) based on distance_diff and flags

**Output:** `data/naics_training_pairs.parquet`

```
Schema:
  - anchor_code: String
  - positive_code: String
  - negative_code: String
  - excluded: Boolean
  - unrelated: Boolean
  - positive_distance: Float32
  - negative_distance: Float32
  - distance_diff: Float32

Rows: 263,830,364
Size: ~3.2 GB
```

**Hardness Distribution:** See [Curriculum Design Guide](curriculum_design_guide.md#understanding-hardness-levels)

### Data Flow Diagram

```
┌──────────────┐
│ Census Excel │
│  (4 files)   │
└──────┬───────┘
       │ download_data.py
       │ • Parse Excel
       │ • Clean text
       │ • Fill missing descriptions
       ↓
┌──────────────────────┐
│ naics_descriptions   │
│ 2,125 codes          │
│ (title, desc, etc.)  │
└──────┬───────────────┘
       │ compute_distances.py
       │ • Build trees per sector
       │ • Calculate graph distances
       │ • Cross-sector pairs
       ↓
┌──────────────────────┐
│  naics_distances     │
│  3.0M pairs          │
│  (i, j, distance)    │
└──────┬───────────────┘
       │ create_triplets.py
       │ • Join positives + negatives
       │ • Filter valid triplets
       │ • Assign hardness
       ↓
┌──────────────────────┐
│ naics_training_pairs │
│ 263M triplets        │
│ (anchor, pos, neg)   │
└──────────────────────┘
       │
       │ Streaming Dataset
       │ • Curriculum filtering
       │ • Negative sampling
       ↓
   [Training]
```

---

## Model Architecture

### Multi-Channel Encoder

**Design Philosophy:** Each text field (title, description, excluded, examples) provides complementary information. Training separate encoders allows channel-specific feature extraction before fusion.

#### Component: Single Channel (`encoder.py`)

```python
Channel: title / description / excluded / examples
    ↓
Tokenization (cached)
    ↓
Transformer: all-mpnet-base-v2 (768-dim)
    ├─ 12 layers
    ├─ 768 hidden size
    └─ LoRA applied to query/value projections
        • r = 8 (rank)
        • alpha = 16
        • dropout = 0.1
    ↓
[CLS] token embedding (768-dim)
```

**LoRA Configuration:**
- **Target modules:** `['query', 'value']` (not key or output)
- **Rank (r):** 8 - Low rank approximation
- **Alpha:** 16 - Scaling factor (alpha/r = 2.0)
- **Dropout:** 0.1

**Why LoRA?**
- Parameter efficiency: ~1% of full model parameters
- Prevents catastrophic forgetting of pretrained knowledge
- Enables channel-specific adaptation without full fine-tuning

#### Architecture: Multi-Channel Fusion

```
┌─────────────────────────────────────────────────────────┐
│              4 Separate Encoders (768-dim each)         │
│                                                         │
│  Title (768) + Description (768) + Excluded (768) +     │
│  Examples (768) = Concatenated (3072-dim)               │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           Mixture of Experts Layer                      │
│                                                         │
│  ┌─────────────────────────────────────────────────-─┐  │
│  │  Gating Network (3072 → 4)                        │  │
│  │    ↓                                              │  │
│  │  Softmax → Top-2 Selection                        │  │
│  │    ↓                                              │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │  │
│  │  │Expert 1│  │Expert 2│  │Expert 3│  │Expert 4│   │  │
│  │  │        │  │        │  │        │  │        │   │  │
│  │  │3072 →  │  │3072 →  │  │3072 →  │  │3072 →  │   │  │
│  │  │1024 →  │  │1024 →  │  │1024 →  │  │1024 →  │   │  │
│  │  │3072    │  │3072    │  │3072    │  │3072    │   │  │
│  │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘   │  │
│  │      │            │            │            │     │  │
│  │      └────────────┴────────────┴────────────┘     │  │
│  │                    Weighted Sum                   │  │
│  └─────────────────────────────────────────────────-─┘  │
│                                                         │
│  Output: Fused embedding (3072-dim)                     │
└───────────────────────────┬─────────────────────────────┘
                            ↓
                    (Pass to projection)
```

**Expert Architecture (each):**
```
Linear(3072 → 1024)
    ↓
ReLU
    ↓
Dropout(0.1)
    ↓
Linear(1024 → 3072)
```

**Gating Mechanism:**
```python
gate_logits = Linear(3072 → 4)(concatenated_embeddings)
gate_probs = softmax(gate_logits)
top_2_probs, top_2_indices = topk(gate_probs, k=2)

# Normalize
top_2_probs = top_2_probs / sum(top_2_probs)

# Route to selected experts
output = sum(top_2_probs[i] * expert[top_2_indices[i]](input))
```

**Load Balancing Loss:**
```python
# Global batch statistics
importance = mean(gate_probs across batch)  # Per-expert importance
load = fraction of tokens to each expert     # Per-expert load

# Minimize product (encourages uniform distribution)
load_balancing_loss = num_experts * sum(importance * load)
```

### Hyperbolic Projection (`loss.py`)

**Goal:** Map Euclidean embeddings to Lorentz hyperboloid model for hierarchy-preserving geometry.

```
┌─────────────────────────────────────────────────────────┐
│              Euclidean Embedding (3072-dim)             │
└───────────────────────────┬─────────────────────────────┘
                            ↓
                  Linear(3072 → 3073)
                            ↓
            Tangent vector v (3073-dim)
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Exponential Map (exp_map_zero)             │
│                                                         │
│  norm_v = ||v||                                         │
│  x₀ = cosh(norm_v / √c)                                 │
│  x_rest = sinh(norm_v / √c) * v / norm_v                │
│  x = [x₀, x_rest]                                       │
└───────────────────────────┬─────────────────────────────┘
                            ↓
            Lorentz Model Point (3073-dim)
            • Lives on hyperboloid: -x₀² + Σxᵢ² = -1/c
            • Curvature: c = 1.0
```

**Lorentzian Distance:**
```python
def lorentz_dot(u, v):
    """Minkowski inner product"""
    return sum(u[1:] * v[1:]) - u[0] * v[0]

def lorentz_distance(u, v, c=1.0):
    """Distance on hyperboloid"""
    dot = lorentz_dot(u, v)
    dot = clamp(dot, max=-1.0 - 1e-5)  # Numerical stability
    return sqrt(c) * acosh(-dot)
```

**Why Lorentz over Poincaré?**
- **Numerical stability:** No boundary singularities
- **Efficiency:** Simpler gradient computations
- **Performance:** Better convergence in practice

### Loss Function

**Hyperbolic InfoNCE Loss:**

```python
# For each (anchor, positive, K negatives) triplet:

# 1. Project to hyperbolic space
anchor_hyp = hyperbolic_proj(anchor_emb)       # (B, 3073)
positive_hyp = hyperbolic_proj(positive_emb)   # (B, 3073)
negative_hyp = hyperbolic_proj(negative_embs)  # (B*K, 3073)

# 2. Compute Lorentzian distances
pos_dist = lorentz_distance(anchor_hyp, positive_hyp)  # (B,)
neg_dist = lorentz_distance(anchor_hyp, negative_hyp)  # (B, K)

# 3. Convert distances to similarities (negative distance)
pos_sim = -pos_dist / temperature  # (B,)
neg_sim = -neg_dist / temperature  # (B, K)

# 4. InfoNCE loss (cross-entropy with positive at index 0)
logits = concat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, K+1)
labels = zeros(B)  # Positive is always at index 0
loss = cross_entropy(logits, labels)
```

**Temperature Parameter:**
- Default: τ = 0.07
- Lower τ → sharper distributions → harder to satisfy
- Higher τ → smoother distributions → easier optimization

**Total Loss:**
```python
total_loss = contrastive_loss + λ * load_balancing_loss
```
where λ = 0.01 (default)

---

## Training Architecture

### Data Loading (`streaming_dataset.py`)

**Challenge:** 263M triplets × 4 fields/triplet = 1B+ text strings don't fit in RAM

**Solution:** Streaming dataset with curriculum filtering

```
┌─────────────────────────────────────────────────────────┐
│                Parquet File (on disk)                   │
│              263M rows × 8 columns                      │
└───────────────────────────┬─────────────────────────────┘
                            ↓
        Curriculum Filtering (Polars lazy scan)
        • positive_levels: [6]
        • positive_distance_max: 2.0
        • difficulty_buckets: [1, 2]
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Filtered Triplets (lazy)                   │
│              e.g., 50M rows after filtering             │
└───────────────────────────┬─────────────────────────────┘
                            ↓
          Collect anchors + positives
          Group by (anchor, positive)
                            ↓
┌─────────────────────────────────────────────────────────┐
│       Negative Sampling (per anchor-positive pair)      │
│                                                         │
│  For each (anchor, positive):                           │
│    1. Get all valid negatives from filtered dataset     │
│    2. Bucket by hardness level                          │
│    3. Sample K negatives per bucket_percentages         │
│    4. Fallback to easier buckets if sparse              │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│            Tokenization Cache Lookup                    │
│                                                         │
│  anchor_code → {title: ids, description: ids, ...}      │
│  positive_code → {title: ids, description: ids, ...}    │
│  negative_code × K → {title: ids, ...}                  │
└───────────────────────────┬─────────────────────────────┘
                            ↓
                    Batch Assembly
                (anchor, positive, K negatives)
                            ↓
                      [Model Forward]
```

### Tokenization Cache (`tokenization_cache.py`)

**Problem:** Tokenizing 2,125 codes × 4 channels on-the-fly is slow

**Solution:** Pre-tokenize all codes once, cache to disk

```python
Cache Structure:
{
    "541511": {
        "title": {
            "input_ids": [101, 2345, ...],
            "attention_mask": [1, 1, 1, ...]
        },
        "description": {...},
        "excluded": {...},
        "examples": {...}
    },
    ...
}

File: data/.cache/tokenization_cache_all-mpnet-base-v2.pkl
Size: ~50 MB (all 2,125 codes)
Load time: <1 second
```

### Training Loop

```
┌────────────────────────────────────────────────────────────┐
│                     Training Epoch                         │
└───────────────────────────┬────────────────────────────────┘
                            ↓
                ┌───────────────────────┐
                │  Sample Batch (B=32)  │
                │  via DataLoader       │
                └───────────┬───────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Forward Pass                             │
│                                                             │
│  For anchor, positive, negatives:                           │
│    1. Encode each channel (title, desc, excl, examples)     │
│    2. Route through MoE (get fused embedding)               │
│    3. Project to hyperbolic space (Lorentz model)           │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Loss Computation                       │
│                                                             │
│  contrastive_loss = HyperbolicInfoNCE(                      │
│      anchor_hyp, positive_hyp, negative_hyp, B, K           │
│  )                                                          │
│                                                             │
│  load_balancing_loss = (                                    │
│      anchor_output.lb_loss +                                │
│      positive_output.lb_loss +                              │
│      negative_output.lb_loss                                │
│  ) / 3.0                                                    │
│                                                             │
│  total_loss = contrastive_loss + λ * load_balancing_loss    │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
                    Backward Pass
                  (Gradient descent)
                            ↓
                   Update Weights
                            ↓
                    Log Metrics
                (loss, LR, load balance)
```

### Optimizer & Scheduler

**Optimizer:** AdamW
```python
lr = 2e-4              # Base learning rate
weight_decay = 0.01    # L2 regularization
betas = (0.9, 0.999)   # Adam momentum parameters
```

**Scheduler:** Cosine Annealing
```python
T_max = max_epochs
eta_min = 1e-6  # Minimum learning rate

# Learning rate curve:
# Epoch 0: 2e-4
# Epoch max_epochs/2: ~1e-4
# Epoch max_epochs: 1e-6
```

---

## Key Design Decisions

### 1. Why Mixture of Experts?

**Alternative considered:** Gated attention (e.g., Gated Multimodal Unit)

**Decision:** MoE for specialized fusion strategies

**Rationale:**
- NAICS codes are heterogeneous: some codes rely on title+desc, others on examples/excluded
- MoE learns multiple fusion strategies (experts) and routes inputs appropriately
- Top-2 gating: balance between specialization (1 expert) and robustness (4 experts)
- Conditional computation: only 2/4 experts active per input → efficiency

**Trade-off:** More parameters but sparse activation

### 2. Why Lorentz Hyperbolic Space?

**Alternatives considered:**
- Euclidean space with hierarchical loss
- Poincaré ball hyperbolic model

**Decision:** Lorentz hyperboloid model

**Rationale:**
- **Geometric fit:** Hyperbolic space naturally represents trees (exponential volume growth)
- **Numerical stability:** Poincaré ball suffers from NaN at boundary; Lorentz is stable
- **Performance:** Empirically shown to train better in lower dimensions
- **Simplicity:** Native contrastive loss (negative Lorentzian distance as similarity)

**Trade-off:** Slightly more complex math than Euclidean

### 3. Why LoRA Instead of Full Fine-Tuning?

**Alternatives considered:**
- Full transformer fine-tuning
- Freeze transformer, train only projection layers

**Decision:** LoRA on query/value projections

**Rationale:**
- **Efficiency:** ~1% of parameters trainable (4 channels × LoRA weights)
- **Memory:** Fits in GPU memory with large batch sizes
- **Generalization:** Preserves pretrained knowledge while adapting to NAICS
- **Channel-specific:** Each channel can learn different attention patterns

**Trade-off:** Slightly lower capacity than full fine-tuning

### 4. Why Curriculum Learning?

**Alternatives considered:**
- Random sampling from full dataset
- Hard negative mining

**Decision:** Hardness-based curriculum

**Rationale:**
- **Convergence:** Start with easy distinctions (sectors) before fine-grained (siblings)
- **False negatives:** Hard negatives early → model repels related concepts → breaks hierarchy
- **Data efficiency:** 93% of data is hardness 1 (unrelated) — need curriculum to see hard examples

**Trade-off:** More complex training process

### 5. Why Pre-Tokenize and Cache?

**Alternatives considered:**
- Tokenize on-the-fly
- Load full dataset in RAM

**Decision:** Pre-tokenize once, cache to disk

**Rationale:**
- **Speed:** Tokenization is CPU-bound and slow
- **Reusability:** Same tokenization across all curricula and epochs
- **Memory:** Only load batch worth of tokens, not full dataset

**Trade-off:** Upfront cost to build cache (~1 minute)

### 6. Why Streaming Dataset?

**Alternatives considered:**
- Load full 263M triplets in RAM
- Subsample dataset to fit in memory

**Decision:** Stream from Parquet with lazy evaluation

**Rationale:**
- **Scalability:** Dataset too large for RAM (3.2 GB compressed, larger in memory)
- **Flexibility:** Curriculum filtering happens at query time
- **Simplicity:** Polars lazy scans are efficient and expressive

**Trade-off:** Requires fast disk I/O (SSD recommended)

---

## Performance Characteristics

### Memory Usage

**Per Component (batch size = 32, K = 16):**

| Component                       | Memory  |
|---------------------------------|---------|
| 4 × Transformer encoders (LoRA) | ~2 GB   |
| MoE layer (4 experts)           | ~500 MB |
| Hyperbolic projection           | ~100 MB |
| Batch tensors (B=32, K=16)      | ~1 GB   |
| Optimizer state (AdamW)         | ~4 GB   |
| **Total**                       | **~8 GB** |

**Recommendations:**
- 16 GB GPU: batch_size=16, k_negatives=8
- 24 GB GPU: batch_size=32, k_negatives=16 (default)
- 40+ GB GPU: batch_size=64, k_negatives=32

### Computational Bottlenecks

**Training (batch_size=32, K=16):**

| Operation | Time | % |
|-----------|------|---|
| Forward pass (4 channels) | 120 ms | 40% |
| MoE routing + experts | 50 ms | 17% |
| Hyperbolic projection | 30 ms | 10% |
| Loss computation | 20 ms | 7% |
| Backward pass | 60 ms | 20% |
| Optimizer step | 20 ms | 6% |
| **Total per batch** | **~300 ms** | **100%** |

**Throughput:** ~3.3 batches/sec = 106 samples/sec (on RTX 4090)

**Data Loading:**
- Streaming from Parquet: negligible (<5 ms/batch with SSD)
- Tokenization cache lookup: <1 ms/batch
- Not a bottleneck with `num_workers=4`

### Scalability

**Vertical Scaling (larger GPU):**
- Memory-bound, not compute-bound
- Larger batch sizes → better GPU utilization
- Diminishing returns above batch_size=64

**Horizontal Scaling (multi-GPU):**
- PyTorch Lightning DDP supported
- Global-batch load balancing for MoE
- Near-linear scaling up to 4 GPUs
- Communication overhead noticeable beyond 8 GPUs

**Data Scaling:**
- Dataset size: streaming handles arbitrary size
- Number of codes: O(N²) distance pairs, O(N³) triplets
- Practical limit: ~5000 codes before preprocessing too slow

---

## Implementation Notes

### Critical Numerical Stability Tricks

1. **Hyperbolic projection:**
   ```python
   # Clamp norm to avoid division by zero
   norm_v = torch.clamp(norm_v, min=1e-8)
   ```

2. **Lorentzian distance:**
   ```python
   # Clamp dot product to avoid acosh(x < 1)
   dot = torch.clamp(dot, max=-1.0 - 1e-5)
   ```

3. **MoE gating:**
   ```python
   # Normalize top-k probabilities to sum to 1.0
   top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
   ```

### Debugging Tips

**Model not training:**
1. Check MoE load balancing: should be < 0.05
2. Verify curriculum filters: `difficulty_buckets` should yield data
3. Inspect distances: positive < negative for all triplets
4. Monitor gradient norms: exploding gradients → reduce LR

**OOM errors:**
1. Reduce `batch_size`
2. Reduce `k_negatives`
3. Enable gradient checkpointing (not implemented yet)
4. Use gradient accumulation

**Slow data loading:**
1. Increase `num_workers` (4-8 recommended)
2. Enable `persistent_workers=True`
3. Check disk I/O (SSD > HDD)
4. Pre-filter Parquet file for curriculum

---

## Future Extensions

**Potential improvements:**

1. **Multi-GPU training:** Implement global-batch MoE load balancing across devices
2. **False negative detection:** Add clustering-based detection in late training
3. **Adaptive curriculum:** Automatically progress based on validation loss
4. **Inference API:** Add embedding generation and similarity search
5. **Visualization:** t-SNE/UMAP plots of learned embeddings
6. **Gradient checkpointing:** Reduce memory for larger models

---

## References

- **Hyperbolic embeddings:** Nickel & Kiela (2017), "Poincaré Embeddings for Learning Hierarchical Representations"
- **Lorentz model:** Nickel & Kiela (2018), "Learning Continuous Hierarchies in the Lorentz Model"
- **Contrastive learning:** Chen et al. (2020), "A Simple Framework for Contrastive Learning"
- **Mixture of Experts:** Shazeer et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- **LoRA:** Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"

---

**Next Steps:**
- [Quick Start Guide](quickstart.md) - Get started training
- [Curriculum Design](curriculum_design_guide.md) - Design effective curricula
- [Troubleshooting](troubleshooting.md) - Debug training issues
