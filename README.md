# NAICS Hyperbolic Embedding System

This project implements a unified hyperbolic representation learning framework for the
**North American Industry Classification System (NAICS)**. The system combines multi-channel text
encoding, Mixture-of-Experts fusion, hyperbolic contrastive learning, and a hyperbolic graph
refinement stage to produce geometry-aware embeddings aligned with the hierarchical structure of
the NAICS taxonomy.

The final output is a set of **Lorentz-model hyperbolic embeddings** suitable for similarity
search, hierarchical modeling, graph-based reasoning, and downstream machine learning applications.

---

## 1. System Architecture Overview

The system consists of four sequential stages:

1. **Multi-channel text encoding** – independent transformer-based encoders for title,
   description, examples, and exclusions.
2. **Mixture-of-Experts (MoE) fusion** – adaptive fusion of the four embeddings using
   Top-2 gating.
3. **Hyperbolic contrastive learning** – projection into Lorentz space and optimization with
   hyperbolic InfoNCE.
4. **Hyperbolic Graph Convolutional Refinement (HGCN)** – structure-aware refinement using the
   explicit NAICS parent–child graph.

Each stage is designed to preserve or enhance the hierarchical geometry of NAICS codes.

---

## 2. Stage 1 — Multi-Channel Text Encoding

Each NAICS code includes four distinct text fields:

- Title
- Description
- Examples
- Excluded codes

Each field is processed independently using a transformer encoder (LoRA-adapted). This produces
four Euclidean embeddings:

- (E_title)
- (E_desc)
- (E_examples)
- (E_excluded)

These embeddings serve as inputs to the fusion stage.

---

## 3. Stage 2 — Mixture-of-Experts Fusion (Top-2 Gating)

The four channel embeddings are concatenated and passed into a **Mixture-of-Experts (MoE)**
module. Key components include:

- Top-2 gating to route each input to the two most relevant experts.
- Feed-forward expert networks that learn specialized fusion behaviors.
- Auxiliary load-balancing loss to ensure even expert utilization across batches.

This produces a single fused Euclidean embedding (E_fused) per NAICS code.

---

## 4. Stage 3 — Hyperbolic Contrastive Learning (Lorentz Model)

To align the latent space with the hierarchical structure of NAICS, embeddings are projected into
**Lorentz-model hyperbolic space** via the exponential map.

### 4.1 Hyperbolic Projection

The fused Euclidean vector is mapped onto the hyperboloid:

- Uses exponential map at the origin
- Supports learned or fixed curvature
- Ensures numerical stability

The result is a Lorentz embedding (E_hyp).

### 4.2 Hyperbolic InfoNCE Loss

Contrastive learning is performed using **Lorentzian geodesic distance**:

d(u, v) = arcosh(-<u, v>_L)

Negatives include:

- unrelated codes,
- hierarchically distant codes,
- false negatives detected via periodic clustering.

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
| (Lorentz Distance + FN Mask)  |
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

## 8. Module API Documentation (Summary)

### 8.1 `MultiChannelEncoder`

**Purpose:** Encodes four NAICS text fields using transformer-based encoders.

**Key Methods:**

- `forward(batch) -> Dict[str, Tensor]`
  - Returns per-channel embeddings and fused embedding.

**Inputs:** tokenized text for each field.  
**Outputs:** Euclidean embeddings prior to hyperbolic projection.

---

### 8.2 `MixtureOfExperts`

**Purpose:** Learns adaptive fusion of the four text channels.

**Key Methods:**

- `forward(x) -> (Tensor, Tensor, Tensor)`
  - Returns fused vector, gating probabilities, and selected experts.

**Notes:**

- Implements Top-2 gating.
- Auxiliary load-balancing loss computed separately.

---

### 8.3 `HyperbolicProjection`

**Purpose:** Maps Euclidean embeddings to the Lorentz hyperboloid.

**Key Methods:**

- `forward(x) -> Tensor`
  - Applies the exponential map.

**Notes:**

- Supports configurable curvature.
- Numerically stable norm clamping.

---

### 8.4 `HyperbolicInfoNCELoss`

**Purpose:** Contrastive learning in hyperbolic space.

**Key Methods:**

- `forward(anchor, pos, neg, mask=None)`
  - Computes Lorentz-based InfoNCE.

**Notes:**

- Integrates false-negative elimination.
- Uses Lorentz geodesic distance.

---

### 8.5 `HGCN` (Graph Refinement)

**Purpose:** Refines embeddings using NAICS parent–child graph.

**Key Components:**

- Hyperbolic graph convolution layers
- Log-/exp-map transitions
- Learnable curvature

**Outputs:** Updated Lorentz-model embeddings.

---

## 9. Onboarding Guide

### 9.1 Training the Contrastive Model

1. Prepare the NAICS dataset with four text channels.
2. Configure the encoder, MoE, and loss parameters.
3. Run training:

   ```bash
   python train.py --config configs/contrastive.yaml
   ```

4. After convergence, extract Lorentz-model embeddings.

### 9.2 Running HGCN Refinement

1. Construct the NAICS parent–child graph.
2. Load hyperbolic embeddings from Stage 3.
3. Train the refinement model:

   ```bash
   python train_hgcn.py --config configs/hgcn.yaml
   ```

4. Save refined embeddings for downstream tasks.

---

## 10. Using the Final Embeddings

### 10.1 Similarity Search

Use Lorentzian distance:

```python
dist = lorentz_distance(x, y)
```

Lower values indicate closer hierarchical or semantic similarity.

### 10.2 Visualization

Project to tangent space or Poincaré ball for plotting.

### 10.3 Downstream ML

Final embeddings can be used as features for:

- classification models,
- clustering algorithms (in hyperbolic or tangent space),
- retrieval and recommendation systems.

---
