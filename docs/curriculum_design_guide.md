# NAICS Gemini Curriculum Design Guide

This guide explains how to design effective curriculum learning strategies for training hierarchical NAICS embeddings. Curriculum learning progressively increases task difficulty, leading to faster convergence and better final model quality.

## Table of Contents

- [Understanding Hardness Levels](#understanding-hardness-levels)
- [Curriculum Configuration](#curriculum-configuration)
- [Design Principles](#design-principles)
- [Built-in Curricula](#built-in-curricula)
- [Creating Custom Curricula](#creating-custom-curricula)
- [Advanced Strategies](#advanced-strategies)
- [Troubleshooting Curricula](#troubleshooting-curricula)

---

## Understanding Hardness Levels

The NAICS training dataset contains 263M+ triplets `(anchor, positive, negative)`, each labeled with a **hardness level** from 1 (easiest) to 8 (hardest). Hardness is determined by semantic similarity and hierarchical distance.

### Hardness Level Definitions

| Level | Type | Distance Diff | Description | Count | % |
|-------|------|---------------|-------------|-------|---|
| **1** | Unrelated | 7.0 | Codes from different sectors (e.g., Agriculture vs Healthcare) | 246.3M | 93.36% |
| **2** | Distant | 4.5-6.5 | Same sector, but distant branches | 379K | 0.14% |
| **3** | Moderate | 3.5-4.0 | Related subsectors | 395K | 0.15% |
| **4** | Related | 2.5-3.0 | Cousins in taxonomy | 827K | 0.31% |
| **5** | Close | 2.0 | Close relatives (e.g., uncle-nephew) | 3.6M | 1.37% |
| **6** | Siblings | 1.0 | Same parent, different codes | 12.2M | 4.61% |
| **7** | Very Hard | 0.5 | Direct parent-child relationships | 4K | 0.00% |
| **8** | Exclusions | N/A | Semantically close but explicitly excluded | 125K | 0.05% |

### Key Concepts

**Distance Difference:**
```
distance_diff = negative_distance - positive_distance
```
- **Low values** (0.5-2.0): Hard negatives — semantically close but still negative
- **High values** (7.0): Easy negatives — completely unrelated

**Unrelated Flag:**
- `unrelated=True`: Negative is from a different sector (always distance_diff = 7.0)
- `unrelated=False`: Negative is within the same sector (distance_diff < 7.0)

**Excluded Flag:**
- `excluded=True`: Negative explicitly mentioned in the positive's "excluded" field
- These are **specially hard** because they're semantically close but industry-distinct
- Example: 541511 (Custom Programming) excludes 541512 (Systems Design)

### Visual Hierarchy Example

```
Sector 54 (Professional Services)
│
├── 541 (Professional, Scientific & Technical Services)
│   │
│   ├── 5415 (Computer Systems Design)
│   │   │
│   │   ├── 54151 (Computer Systems Design)
│   │   │   ├── 541511 (Custom Programming) ← ANCHOR
│   │   │   │   excludes: 541512 (exclusion)
│   │   │   │
│   │   │   └── 541512 (Systems Design) ← Hardness 8 (excluded)
│   │   │       
│   │   └── 54152 (Other Design Services) ← Hardness 6 (sibling)
│   │
│   └── 5416 (Management Consulting) ← Hardness 4-5 (cousin)
│
└── 5419 (Other Professional Services) ← Hardness 2-3 (distant)

Sector 62 (Healthcare) ← Hardness 1 (unrelated)
```

---

## Curriculum Configuration

Curricula are defined in YAML files under `conf/curriculum/`. Each curriculum filters the 263M triplet dataset and controls negative sampling.

### Configuration Parameters

```yaml
name: my_curriculum_name

# ============================================================================
# Positive Pair Filtering
# ============================================================================

positive_levels: [6]
# Which NAICS hierarchy levels to include for positive pairs
# Options: [2, 3, 4, 5, 6] or any subset
# - Level 2: Sector codes (e.g., 11, 54)
# - Level 6: Full 6-digit codes (e.g., 541511)
# Recommendation: Start with [6] for specificity

positive_distance_min: null
# Minimum graph distance between anchor and positive
# null = no minimum
# Example: 1.0 = exclude direct parent-child (lineal) relationships

positive_distance_max: 2.0
# Maximum graph distance between anchor and positive
# null = no maximum
# Example: 2.0 = only include close relatives and siblings

max_positives: null
# Cap on number of positives per anchor
# null = include all matching positives
# Example: 10 = sample up to 10 positives per anchor

# ============================================================================
# Negative Sampling
# ============================================================================

difficulty_buckets: [1, 2]
# Which hardness levels to sample negatives from
# Options: [1, 2, 3, 4, 5, 6, 7, 8]
# Recommendation: Start with [1, 2] (unrelated + distant)

bucket_percentages:
  1: 0.875  # 87.5% unrelated negatives
  2: 0.125  # 12.5% distant negatives
# Must sum to 1.0
# Defines composition of K negatives per positive

k_negatives: 16
# Total number of negatives per (anchor, positive) pair
# Higher K = more contrastive signal but slower training
# Recommendation: 16-32
```

---

## Design Principles

### 1. Start Easy, Progress Gradually

**Principle:** Begin with high-level distinctions before fine-grained nuances.

**Rationale:** 
- Embedding space is random at initialization
- Model needs stable gradients to form semantic structure
- Hard negatives early → conflicting signals → slow convergence

**Implementation:**
```yaml
# Stage 1: Learn sector boundaries
positive_levels: [6]
difficulty_buckets: [1]  # Only unrelated negatives
bucket_percentages: {1: 1.0}

# Stage 2: Add within-sector distinctions
difficulty_buckets: [1, 2]
bucket_percentages: {1: 0.75, 2: 0.25}

# Stage 3: Include hard negatives (siblings)
difficulty_buckets: [1, 2, 5, 6]
bucket_percentages: {1: 0.5, 2: 0.2, 5: 0.2, 6: 0.1}
```

### 2. Balance Positive Distance Constraints

**Principle:** Control how similar your positive pairs are.

**Trade-offs:**
- **Tight constraints** (e.g., `max: 1.0`): Model learns fine distinctions
  - Risk: Limited training data, may overfit
- **Loose constraints** (e.g., `max: 4.0`): More training data, broader concepts
  - Risk: Positive pairs too dissimilar, conflicts with hard negatives

**Recommendation:**
```yaml
# Conservative (recommended for early training)
positive_distance_max: 2.0  # Siblings + close relatives

# Aggressive (for later refinement)
positive_distance_max: null  # All positives within sector
```

### 3. Allocate Bucket Percentages Strategically

**Principle:** Reflect the natural distribution but bias toward learning objectives.

**Natural distribution:**
- 93% of triplets are hardness 1 (unrelated)
- Only 0.05% are hardness 8 (exclusions)

**Strategy:**
- **Early training:** Over-weight unrelated (80-100%) to establish sectors
- **Mid training:** Introduce moderate difficulty (20-30%)
- **Late training:** Small percentage of hard negatives (5-15%)

**Anti-pattern:** Equal weighting across all buckets
```yaml
# ❌ BAD: Forces rare hard negatives
bucket_percentages: {1: 0.33, 6: 0.33, 8: 0.34}  # Don't do this!

# ✅ GOOD: Natural progression
bucket_percentages: {1: 0.60, 2: 0.25, 5: 0.10, 6: 0.05}
```

### 4. Handle Exclusions with Care

**Principle:** Hardness 8 (exclusions) are **extremely difficult** — use sparingly.

**Why exclusions are special:**
- Semantically very close (e.g., both are "computer services")
- Industry distinctions are subtle (programming vs systems design)
- Only 125K triplets available (sparse)

**Recommendation:**
```yaml
# ❌ Too early — model not ready
# Stage 1:
difficulty_buckets: [1, 8]  # Don't include exclusions yet!

# ✅ Final stage only
# Stage 3:
difficulty_buckets: [1, 2, 5, 6, 8]
bucket_percentages: {1: 0.50, 2: 0.20, 5: 0.15, 6: 0.10, 8: 0.05}
```

### 5. Ensure Sufficient K Negatives

**Principle:** Every positive must have K negatives, even if buckets are sparse.

**Fallback mechanism:**
If a hardness level has insufficient triplets, the sampler falls back to easier levels:
```
Requested: Hardness 7 (4K triplets) 
→ Falls back to: Hardness 6 (12M triplets)
→ Falls back to: Hardness 5 (3.6M triplets)
→ Eventually: Hardness 1 (246M triplets)
```

**Implication:** Percentages are **targets**, not guarantees.

**Recommendation:**
```yaml
k_negatives: 16  # Reasonable default

# If you have abundant data for your buckets:
k_negatives: 32  # More contrastive signal

# If buckets are very sparse:
k_negatives: 8   # Reduce demand
```

---

## Built-in Curricula

### Stage 1: Easy (`01_stage_easy.yaml`)

**Goal:** Learn to distinguish sectors (unrelated codes).

```yaml
name: 01_stage_easy

positive_levels: [6]           # Full 6-digit codes
positive_distance_max: 2.0     # Siblings and close relatives only
max_positives: null

difficulty_buckets: [1, 2]     # Unrelated + distant
bucket_percentages:
  1: 0.875  # 87.5% unrelated (different sectors)
  2: 0.125  # 12.5% distant (same sector, far apart)

k_negatives: 16
```

**When to use:**
- First training run
- After architecture changes
- Debugging convergence issues

**Expected outcome:**
- Model learns sector boundaries
- Embeddings cluster by sector (11, 21, 54, 62, etc.)
- Validation loss: ~2.0-2.5 after 5 epochs

### Stage 2: Medium (`02_stage_medium.yaml`)

**Goal:** Refine within-sector distinctions, introduce hard negatives.

```yaml
name: 02_stage_medium

positive_levels: [5, 6]        # 5 and 6-digit codes
positive_distance_max: 3.0     # Include cousins
max_positives: null

difficulty_buckets: [1, 2, 5, 6]  # Add close relatives and siblings
bucket_percentages:
  1: 0.50   # 50% unrelated
  2: 0.25   # 25% distant
  5: 0.15   # 15% close relatives
  6: 0.10   # 10% siblings

k_negatives: 16
```

**When to use:**
- After Stage 1 converges
- When sector boundaries are stable
- Before final refinement

**Expected outcome:**
- Model learns subsector distinctions
- Siblings are pushed apart but remain closer than unrelated codes
- Validation loss: ~1.5-2.0

### Stage 3: Hard (`03_stage_hard.yaml`)

**Goal:** Learn fine-grained distinctions, handle exclusions.

```yaml
name: 03_stage_hard

positive_levels: [2, 3, 4, 5, 6]  # All levels
positive_distance_max: null        # All positives
max_positives: null

difficulty_buckets: [1, 2, 3, 4, 5, 6, 8]  # Include exclusions
bucket_percentages:
  1: 0.40   # 40% unrelated
  2: 0.20   # 20% distant
  3: 0.10   # 10% moderate
  4: 0.10   # 10% related
  5: 0.10   # 10% close
  6: 0.05   # 5% siblings
  8: 0.05   # 5% exclusions (very hard!)

k_negatives: 16
```

**When to use:**
- Final training stage
- After Stage 2 converges
- For production models

**Expected outcome:**
- Model learns subtle distinctions
- Exclusions are correctly separated
- Hierarchical structure is preserved end-to-end
- Validation loss: ~1.0-1.5

---

## Creating Custom Curricula

### Step 1: Copy a Template

```bash
cp conf/curriculum/01_stage_easy.yaml conf/curriculum/my_experiment.yaml
```

### Step 2: Edit Configuration

```yaml
name: my_experiment

# Example: Focus on manufacturing sector (codes 31-33)
# Want to learn fine-grained distinctions within manufacturing only

positive_levels: [6]
positive_distance_max: 2.0     # Siblings only
max_positives: 5               # Sample 5 positives per anchor

difficulty_buckets: [1, 5, 6]  # Unrelated + close + siblings
bucket_percentages:
  1: 0.50   # Half unrelated (preserve sector boundaries)
  5: 0.30   # Close relatives within manufacturing
  6: 0.20   # Siblings

k_negatives: 24  # More negatives for richer signal
```

### Step 3: Train

```bash
uv run naics-gemini train --curriculum my_experiment
```

### Step 4: Monitor and Iterate

Watch key metrics:
- **Contrastive loss:** Should decrease steadily
- **Load balancing:** Should stay < 0.05
- **Validation loss:** Should track training loss

If issues arise, see [Troubleshooting Curricula](#troubleshooting-curricula).

---

## Advanced Strategies

### Strategy 1: Multi-Stage Progression

Train sequentially with increasing difficulty:

```bash
# Stage 1: Warm-up (3 epochs)
uv run naics-gemini train -c 01_stage_easy \
  training.trainer.max_epochs=3

# Stage 2: Load checkpoint, continue with harder curriculum (5 epochs)
uv run naics-gemini train -c 02_stage_medium \
  training.trainer.max_epochs=5 \
  model.checkpoint_path=checkpoints/01_stage_easy/last.ckpt

# Stage 3: Final refinement (10 epochs)
uv run naics-gemini train -c 03_stage_hard \
  training.trainer.max_epochs=10 \
  model.checkpoint_path=checkpoints/02_stage_medium/last.ckpt
```

### Strategy 2: Adaptive Curriculum

Dynamically adjust bucket percentages based on validation performance:

```yaml
# Week 1: Heavy unrelated emphasis
bucket_percentages: {1: 0.90, 2: 0.10}

# Week 2: If val loss < 2.0, increase difficulty
bucket_percentages: {1: 0.70, 2: 0.20, 5: 0.10}

# Week 3: If val loss < 1.5, add hard negatives
bucket_percentages: {1: 0.50, 2: 0.25, 5: 0.15, 6: 0.10}
```

### Strategy 3: Level-Specific Curricula

Train separate models for different hierarchy levels:

```yaml
# Model A: Sector-level (2-digit)
positive_levels: [2]
difficulty_buckets: [1]  # Only needs to distinguish sectors

# Model B: Industry-level (3-4 digit)
positive_levels: [3, 4]
difficulty_buckets: [1, 2, 3]

# Model C: Fine-grained (5-6 digit)
positive_levels: [5, 6]
difficulty_buckets: [1, 2, 5, 6, 8]  # Full difficulty spectrum
```

### Strategy 4: Exclusion-Focused Training

Create a curriculum specifically to learn exclusion distinctions:

```yaml
name: exclusions_only

positive_levels: [6]
positive_distance_max: 1.0  # Tight constraint

difficulty_buckets: [8]     # Only exclusions!
bucket_percentages: {8: 1.0}

k_negatives: 8  # Fewer negatives (sparse data)

# Note: This is VERY HARD. Only use after general training.
```

---

## Troubleshooting Curricula

### Issue 1: Loss Not Decreasing

**Symptoms:**
- Flat training loss after several epochs
- Loss starts high (~5.0+) and stays there

**Possible causes:**

1. **Curriculum too hard too soon**
   ```yaml
   # ❌ Problem:
   difficulty_buckets: [6, 7, 8]  # All hard negatives from start
   
   # ✅ Solution:
   difficulty_buckets: [1, 2]     # Start easier
   ```

2. **Insufficient positives**
   ```yaml
   # ❌ Problem:
   positive_distance_max: 0.5  # Too restrictive
   
   # ✅ Solution:
   positive_distance_max: 2.0  # More positives available
   ```

3. **Learning rate too high**
   ```bash
   # Try lower LR
   uv run naics-gemini train -c my_curriculum \
     training.learning_rate=5e-5
   ```

### Issue 2: Validation Loss Diverging

**Symptoms:**
- Training loss decreases
- Validation loss increases or plateaus

**Diagnosis:** Model overfitting to curriculum constraints

**Solutions:**

1. **Broaden positive constraints**
   ```yaml
   # Before:
   positive_levels: [6]
   positive_distance_max: 1.0
   
   # After:
   positive_levels: [5, 6]
   positive_distance_max: 3.0
   ```

2. **Increase bucket diversity**
   ```yaml
   # Before:
   difficulty_buckets: [1]
   
   # After:
   difficulty_buckets: [1, 2, 5]
   bucket_percentages: {1: 0.70, 2: 0.20, 5: 0.10}
   ```

### Issue 3: MoE Mode Collapse

**Symptoms:**
- `load_balancing_loss > 0.2`
- Only 1-2 experts being used

**Cause:** Curriculum too uniform — all examples look the same

**Solutions:**

1. **Increase bucket diversity**
   ```yaml
   # Add more hardness levels
   difficulty_buckets: [1, 2, 3, 5, 6]
   ```

2. **Increase load balancing coefficient**
   ```bash
   uv run naics-gemini train -c my_curriculum \
     model.moe.load_balancing_coef=0.02
   ```

### Issue 4: K Negatives Not Sampled

**Symptoms:**
- Console warnings: "Insufficient negatives for hardness X"
- Actual K < requested K

**Cause:** Sparse hardness levels (especially 7, 8)

**Solutions:**

1. **Reduce K**
   ```yaml
   k_negatives: 8  # Instead of 16
   ```

2. **Remove sparse buckets**
   ```yaml
   # ❌ Problem:
   difficulty_buckets: [7, 8]  # Only 129K total triplets
   
   # ✅ Solution:
   difficulty_buckets: [1, 2, 5, 6]  # 262M triplets
   ```

3. **Adjust percentages**
   ```yaml
   # Reduce allocation to sparse levels
   bucket_percentages:
     1: 0.60
     2: 0.25
     8: 0.05  # Was 0.15, now reduced
   ```

### Issue 5: Model Memorizing Curricula

**Symptoms:**
- Perfect training loss
- Poor generalization to new curricula
- Can't distinguish concepts outside trained buckets

**Solution:** Final stage should be diverse

```yaml
name: 03_stage_hard_comprehensive

positive_levels: [2, 3, 4, 5, 6]  # All levels
positive_distance_max: null        # All distances

difficulty_buckets: [1, 2, 3, 4, 5, 6, 8]  # All hardness
bucket_percentages:
  1: 0.40
  2: 0.20
  3: 0.10
  4: 0.10
  5: 0.10
  6: 0.05
  8: 0.05

k_negatives: 24  # Higher K for final training
```

---

## Best Practices Summary

✅ **DO:**
- Start with unrelated negatives (hardness 1)
- Progress gradually through stages
- Monitor load balancing loss
- Use built-in curricula as templates
- Allocate bucket percentages based on learning objectives
- Test curriculum on small subset before full training

❌ **DON'T:**
- Include hardness 8 (exclusions) in early training
- Use equal percentages across all buckets
- Set `positive_distance_max` too restrictively
- Change multiple parameters simultaneously
- Ignore validation loss trends
- Skip Stage 1 (easy curriculum)

---

## Curriculum Planning Template

Use this template to plan your training progression:

```yaml
# Training Plan: [Your Project Name]
# Target: [Your Goal, e.g., "Manufacturing sector embeddings"]
# Duration: [Expected training time]

# ========================================
# Stage 1: Foundation (X epochs)
# ========================================
# Goal: Learn sector boundaries
# Expected val loss: 2.0-2.5

positive_levels: [6]
positive_distance_max: 2.0
difficulty_buckets: [1]
bucket_percentages: {1: 1.0}
k_negatives: 16

# ========================================
# Stage 2: Refinement (Y epochs)
# ========================================
# Goal: Within-sector distinctions
# Expected val loss: 1.5-2.0

positive_levels: [5, 6]
positive_distance_max: 3.0
difficulty_buckets: [1, 2, 5]
bucket_percentages: {1: 0.60, 2: 0.25, 5: 0.15}
k_negatives: 16

# ========================================
# Stage 3: Fine-tuning (Z epochs)
# ========================================
# Goal: Hard negatives + exclusions
# Expected val loss: 1.0-1.5

positive_levels: [2, 3, 4, 5, 6]
positive_distance_max: null
difficulty_buckets: [1, 2, 5, 6, 8]
bucket_percentages: {1: 0.50, 2: 0.20, 5: 0.15, 6: 0.10, 8: 0.05}
k_negatives: 24
```

---

**Next Steps:**
- Try the built-in curricula: [Quick Start](quickstart.md#training-your-first-model)
- Understand the architecture: [Architecture Guide](architecture.md)
- Debug training issues: [Troubleshooting Guide](troubleshooting.md)
