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
