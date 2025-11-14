# NAICS Hyperbolic Embedding System — Contrastive Training Guide

## Overview

This document describes how to train the contrastive representation learning model that forms the
first three stages of the NAICS Hyperbolic Embedding System. The objective of this stage is to
produce Lorentz-model hyperbolic embeddings that capture both semantic and hierarchical
relationships among NAICS codes.

## 1. Training Pipeline

The contrastive training procedure includes:

- Multi-channel transformer encoding for each NAICS text field
- Mixture-of-Experts (MoE) fusion with Top-2 gating
- Hyperbolic projection using the Lorentz exponential map
- Hyperbolic InfoNCE loss using Lorentzian distance
- False-negative mitigation using clustering-based pseudo-labels

## 2. Running the Training Script

```bash
python train.py --config configs/contrastive.yaml
```

## 3. Multi-Channel Encoding

Each NAICS code provides four text channels:

- Title
- Description
- Examples
- Excluded codes

## 4. Mixture-of-Experts Fusion

The MoE module performs conditional fusion of the text channels.

## 5. Hyperbolic Projection (Lorentz Model)

The fused Euclidean vector is projected into the Lorentz model using the exponential map.

## 6. Hyperbolic Contrastive Loss (InfoNCE)

Contrastive learning uses Lorentzian geodesic distance.

## 7. False-Negative Mitigation

Late-training clustering identifies semantically similar negatives to exclude.

## 8. Output of Training

Produces Lorentz hyperbolic embeddings with global semantic structure.
