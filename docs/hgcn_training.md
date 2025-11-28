# NAICS Hyperbolic Embedding System — HGCN Refinement Guide

## Overview

This document explains the final stage of the NAICS hyperbolic embedding pipeline: refinement
using a Hyperbolic Graph Convolutional Network (HGCN).

## 1. Purpose of HGCN Refinement

Integrates NAICS taxonomy directly into embedding geometry.

## 2. Input Requirements

- Lorentz hyperbolic embeddings
- NAICS parent–child graph
- Level metadata

## 3. Running the Refinement

```bash
python train_hgcn.py --config configs/hgcn.yaml
```

## 4. HGCN Layer Operation

Each layer performs log-map, graph convolution in tangent space, activation, and exp-map.

## 5. Refinement Loss Functions

- Hyperbolic Triplet Loss
- Per-Level Radial Regularization

## 6. Learnable Curvature

Curvature parameter is optimized jointly.

## 7. Output of HGCN Refinement

Refined Lorentz-model hyperbolic embeddings aligned with taxonomy structure.

## 8. Validation Metrics

Stage 4 now mirrors the text-model evaluation suite so you can verify that graph refinement does not erode global structure:

- **Cophenetic correlation** – correlation between embedding distances and tree distances.
- **Spearman correlation** – rank-order agreement across the hierarchy.
- **NDCG@K (default: 5/10/20)** – position-aware ranking quality.
- **Distortion stats** – mean/std/median stretch between embedding and tree distances.

Metrics are logged once per validation run (default: every epoch). They require the precomputed tree distance matrix produced in Stage 2.

### Configuration

Add the following keys to your `GraphConfig` (or `configs/hgcn.yaml`) to customize evaluation:

| Key | Description |
| --- | --- |
| `distance_matrix_parquet` | Path to `naics_distance_matrix.parquet`. Required to unlock hierarchy metrics. |
| `full_eval_frequency` | Run the expensive metrics every _N_ optimizer steps (default `1`, meaning every validation epoch). |
| `ndcg_k_values` | List of K values used for NDCG logging. |

If `distance_matrix_parquet` is missing, HGCN automatically skips the extra metrics and continues with the lightweight batch metrics (triplet accuracy, etc.).

## 9. Pre/Post Verification Workflow

After both Stage 3 and Stage 4 finish, run the automated comparison from [Issue #67](https://github.com/lowmason/naics-embedder/issues/67) to confirm that HGCN preserved the Stage 3 geometry:

```bash
uv run naics-embedder tools verify-stage4 \
  --pre ./output/hyperbolic_projection/encodings.parquet \
  --post ./output/hgcn/encodings.parquet
```

Additional options let you override the distance matrix, relations parquet, or the acceptable degradation thresholds:

| Option | Purpose |
| --- | --- |
| `--max-cophenetic-drop` | Maximum allowable decrease in cophenetic correlation (default `0.02`). |
| `--max-ndcg-drop` | Maximum allowable decrease in NDCG@K (default `0.01`). |
| `--min-local-improvement` | Required increase in parent retrieval accuracy (default `0.05`). |
| `--ndcg-k` | Which `K` to evaluate for NDCG (default `10`). |
| `--parent-top-k` | Size of the neighborhood used for parent retrieval (default `1`). |

The command prints pre/post metrics, deltas, and PASS/FAIL indicators for each threshold. Integrate it into CI to prevent regressions before shipping updated embeddings.
