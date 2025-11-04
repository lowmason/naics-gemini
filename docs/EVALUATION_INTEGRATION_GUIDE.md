# 📊 Evaluation Metrics Integration Guide

This guide explains how the evaluation metrics have been integrated into your NAICS Gemini training loop.

## 🎯 What Was Added

### 1. **evaluation.py** (NEW FILE)
Location: `src/naics_gemini/model/evaluation.py`

Contains 5 PyTorch-based evaluation classes:
- `EmbeddingEvaluator` - Distance and similarity computations
- `RetrievalMetrics` - Precision@k, Recall@k, MAP, NDCG@k
- `HierarchyMetrics` - Cophenetic correlation, Spearman correlation, distortion
- `EmbeddingStatistics` - Embedding space analysis and collapse detection
- `NAICSEvaluationRunner` - Complete evaluation pipeline

### 2. **Updated naics_model.py**
Location: `src/naics_gemini/model/naics_model.py`

**New Features:**
- Loads ground truth NAICS distances on initialization
- Caches validation embeddings during validation steps
- Runs comprehensive evaluation at the end of each validation epoch
- Logs all metrics to TensorBoard

**New Parameters:**
```python
NAICSContrastiveModel(
    # ... existing parameters ...
    distances_path='./data/naics_distances.parquet',  # Path to ground truth
    eval_every_n_epochs=1,                             # Evaluate every N epochs
    eval_sample_size=500                               # Max codes to evaluate (for speed)
)
```

### 3. **Updated cli.py**
Location: `src/naics_gemini/cli.py`

- Automatically passes `distances_path` to model
- Shows evaluation metrics status on startup
- No changes needed to existing commands

### 4. **Updated datamodule.py**
Location: `src/naics_gemini/data_loader/datamodule.py`

- Includes NAICS codes in batches for tracking
- Allows model to map embeddings to specific codes during validation

## 📈 Metrics Logged to TensorBoard

### Training Metrics (Every Step)
- `train/contrastive_loss` - Main contrastive loss
- `train/load_balancing_loss` - MoE load balancing
- `train/total_loss` - Combined loss

### Validation Metrics (Every Step)
- `val/contrastive_loss` - Validation loss

### Evaluation Metrics (Every Epoch)
**Hierarchy Preservation:**
- `val/cophenetic_correlation` ⭐ - How well embeddings preserve tree structure (higher is better)
- `val/spearman_correlation` - Rank-based correlation with ground truth

**Embedding Quality:**
- `val/mean_norm` - Average embedding magnitude
- `val/std_norm` - Variation in embedding magnitudes
- `val/mean_pairwise_distance` - Average distance between embeddings
- `val/std_embedding` - Average standard deviation across dimensions

**Collapse Detection:**
- `val/variance_collapsed` - Whether embeddings have low variance (0=no, 1=yes)
- `val/norm_collapsed` - Whether all norms are too similar
- `val/distance_collapsed` - Whether all distances are too similar

**Distortion:**
- `val/mean_distortion` - Average stretch/compression factor
- `val/std_distortion` - Variation in distortion

## 🚀 How to Use

### Step 1: Install Updated Files

Copy the 4 updated files to your project:

```bash
# Copy evaluation.py (NEW)
cp /path/to/outputs/evaluation.py src/naics_gemini/model/evaluation.py

# Copy updated files
cp /path/to/outputs/naics_model.py src/naics_gemini/model/naics_model.py
cp /path/to/outputs/cli.py src/naics_gemini/cli.py
cp /path/to/outputs/datamodule.py src/naics_gemini/data_loader/datamodule.py
```

### Step 2: Run Training (No Changes Required!)

```bash
# Train as usual - evaluation is automatic
uv run naics-gemini train --curriculum 01_stage_easy
```

### Step 3: Monitor in TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/01_stage_easy

# Open browser to http://localhost:6006
```

## 📊 Interpreting Metrics

### Key Metrics to Watch

**1. Cophenetic Correlation** (`val/cophenetic_correlation`)
- **Range:** -1 to 1
- **Goal:** > 0.7 (good hierarchy preservation)
- **What it means:** How well embedding distances match tree distances
- **If too low:** Model not learning hierarchical structure

**2. Collapse Detection** (`val/*_collapsed`)
- **Range:** 0 (good) or 1 (bad)
- **Goal:** All should be 0
- **What it means:** Whether embeddings have collapsed to similar values
- **If any are 1:** Model mode collapse - adjust learning rate or architecture

**3. Mean Distortion** (`val/mean_distortion`)
- **Range:** ~0.5 to 2.0 (typical)
- **Goal:** Close to 1.0
- **What it means:** Average ratio of embedding distance to tree distance
- **< 1:** Embeddings compressed (too close)
- **> 1:** Embeddings stretched (too far)

### Monitoring Convergence

**Healthy Training:**
```
Epoch 1:  cophenetic=0.45, mean_distortion=1.5, collapse=False
Epoch 5:  cophenetic=0.65, mean_distortion=1.2, collapse=False
Epoch 10: cophenetic=0.75, mean_distortion=1.0, collapse=False  ✓ Good!
```

**Mode Collapse:**
```
Epoch 1:  cophenetic=0.45, collapse=False
Epoch 5:  cophenetic=0.30, collapse=True   ✗ Warning!
Epoch 10: cophenetic=0.15, collapse=True   ✗ Bad - restart with lower LR
```

**Poor Hierarchy Learning:**
```
Epoch 1:  cophenetic=0.25, mean_distortion=2.5
Epoch 5:  cophenetic=0.30, mean_distortion=2.3
Epoch 10: cophenetic=0.35, mean_distortion=2.0  ✗ Not improving enough
```

## 🎛️ Configuration Options

### Adjust Evaluation Frequency

```bash
# Evaluate every 2 epochs instead of every epoch (faster)
uv run naics-gemini train -c 01_stage_easy \
  model.eval_every_n_epochs=2
```

### Adjust Sample Size

```bash
# Evaluate on more codes (slower but more accurate)
uv run naics-gemini train -c 01_stage_easy \
  model.eval_sample_size=1000

# Evaluate on fewer codes (faster)
uv run naics-gemini train -c 01_stage_easy \
  model.eval_sample_size=200
```

### Disable Evaluation

If you want to disable evaluation temporarily:

```python
# In naics_model.py __init__, set:
eval_every_n_epochs=1000  # Effectively disables it
```

## 🔧 Performance Impact

**Evaluation overhead:**
- ~10-30 seconds per epoch (with 500 codes)
- Scales linearly with `eval_sample_size`
- Only runs at epoch end (doesn't slow down training steps)

**Memory usage:**
- Minimal (~100MB for cached embeddings)
- Automatically cleared after each epoch

**Recommendations:**
- Keep `eval_sample_size=500` for fast feedback
- Use `eval_every_n_epochs=1` during development
- Increase to `eval_every_n_epochs=5` for long production runs

## 📉 Troubleshooting

### Issue: "Missing embeddings or ground truth distances"

**Cause:** Distance file not found

**Solution:**
```bash
# Generate distances file
uv run naics-gemini data distances
```

### Issue: Evaluation taking too long

**Solution:**
```bash
# Reduce sample size
uv run naics-gemini train -c 01_stage_easy \
  model.eval_sample_size=200
```

### Issue: OOM during evaluation

**Solution:**
```bash
# Reduce sample size or disable evaluation
uv run naics-gemini train -c 01_stage_easy \
  model.eval_sample_size=100
```

### Issue: Metrics not showing in TensorBoard

**Cause:** TensorBoard not refreshed or wrong directory

**Solution:**
```bash
# Restart TensorBoard
tensorboard --logdir outputs/01_stage_easy --reload_interval=5
```

## 🎨 TensorBoard Tips

### Best Views

**1. Scalars Tab:**
- Compare `val/cophenetic_correlation` and `val/contrastive_loss`
- Watch for inverse relationship (loss down → correlation up)

**2. Smoothing:**
- Set smoothing to 0.6 for cleaner trends

**3. Regex Filtering:**
- `val/.*correlation` - Show all correlation metrics
- `val/.*collapsed` - Show collapse detection
- `train/.*` - Show only training metrics

## 🚀 Next Steps

1. **Install the updated files** (see Step 1 above)
2. **Run training** with evaluation enabled
3. **Open TensorBoard** to monitor metrics
4. **Watch cophenetic correlation** - aim for > 0.7
5. **Check collapse detection** - should stay at 0
6. **Adjust hyperparameters** based on metrics

## 📝 Example Training Session

```bash
# Generate data (if not already done)
uv run naics-gemini data all

# Train with evaluation
uv run naics-gemini train --curriculum 01_stage_easy

# In another terminal, start TensorBoard
tensorboard --logdir outputs/01_stage_easy

# Open http://localhost:6006 and watch:
# - val/cophenetic_correlation (should increase)
# - val/contrastive_loss (should decrease)
# - val/*_collapsed (should stay at 0)
```

## ✅ Success Criteria

Your model is training well if:
- ✓ `val/cophenetic_correlation` increases steadily and reaches > 0.7
- ✓ `val/contrastive_loss` decreases steadily
- ✓ All `val/*_collapsed` metrics stay at 0
- ✓ `val/mean_distortion` approaches 1.0
- ✓ `val/spearman_correlation` > 0.6

## 🎓 Understanding the Metrics

### Cophenetic Correlation
- Measures how well your embedding distances correlate with tree distances
- Perfect score = 1.0 (exact match)
- Score > 0.7 = good hierarchy preservation
- Score < 0.5 = poor hierarchy learning

### Distortion
- Ratio of embedding distance to tree distance
- Mean = 1.0 is ideal (no stretch or compression)
- Std should be low (consistent across all pairs)

### Collapse Detection
- Critical for catching mode collapse early
- If any collapse metric = 1, stop training and adjust hyperparameters
- Common fixes: reduce learning rate, increase MoE load balancing

---

**Questions?** Check the [Troubleshooting Guide](troubleshooting.md) or open an issue!
