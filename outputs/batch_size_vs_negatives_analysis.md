# Batch Size vs Negatives: Analysis & Recommendation

## Current Configuration

- **Batch Size**: 12
- **Gradient Accumulation**: 2 (effective batch size: 24)
- **Negatives per Anchor**: 8-48 (varies by stage)
- **Total Negatives per Batch**: 96-576 (batch_size × n_negatives)

## Understanding InfoNCE Loss Mechanics

In your implementation, InfoNCE creates a **classification problem** for each anchor:
- For each anchor, the model must identify the positive among (1 positive + k_negatives)
- This is a (k_negatives + 1)-way classification problem per anchor
- The loss is computed independently for each anchor in the batch

### Key Insight
The **difficulty** of each anchor's task depends on `k_negatives`, not batch size. However, batch size affects:
- Diversity of anchors seen per update
- Gradient variance
- Memory usage

## Research Findings

### 1. **More Negatives Generally Better (with diminishing returns)**
- **Hard negatives**: More negatives increase the chance of sampling hard negatives, which provide better learning signal
- **Gradient quality**: More negatives give clearer signal about what "not similar" means
- **Diminishing returns**: Beyond ~16-32 negatives, improvements are marginal
- **Memory cost**: Linear increase with number of negatives

### 2. **Larger Batches Help But Less Critical**
- **Diversity**: More anchors per batch = more diverse samples per gradient update
- **Gradient stability**: Larger batches reduce gradient variance
- **Memory cost**: Linear increase with batch size
- **For InfoNCE**: Batch size doesn't directly affect the difficulty of each anchor's task

### 3. **Trade-off Analysis**

**Memory Usage:**
- More negatives: `O(batch_size × n_negatives × embedding_dim)`
- More batches: `O(batch_size × embedding_dim)` (but more forward passes)

**Learning Signal:**
- More negatives: Better hard negative mining, clearer "not similar" signal
- More batches: More diverse samples, better coverage of data distribution

## Recommendation for Your Model

### **Prioritize More Negatives (with constraints)**

**Reasoning:**

1. **Your Current Setup is Negatives-Limited**
   - Batch size: 12 (small but manageable)
   - Negatives: 8-48 (varies by stage)
   - **Current total negatives per batch: 96-576**
   
2. **InfoNCE Benefits More from Hard Negatives**
   - Your model uses curriculum learning with increasing difficulty
   - More negatives = better hard negative mining
   - Better signal for hierarchy preservation

3. **Memory Efficiency**
   - Your batch size is already constrained by GPU memory
   - Increasing negatives uses memory more efficiently than increasing batch size
   - Negatives are batched efficiently in your implementation

4. **Your Specific Use Case**
   - **Hierarchical structure**: More negatives help model learn fine-grained distinctions
   - **Hyperbolic space**: More negatives provide better coverage of the embedding space
   - **Curriculum learning**: Stages already increase negatives (8→24→28), suggesting this is the right direction

### Optimal Configuration

**Recommended:**
- **Keep batch size**: 12 (or slightly increase to 16 if memory allows)
- **Increase negatives**: 
  - Stage 1: 16-24 negatives (currently 24, good)
  - Stage 2: 32-48 negatives (currently 24, could increase)
  - Stage 3+: 32-64 negatives (currently 28, could increase)

**Rationale:**
- Total negatives per batch: 192-768 (12 × 16 to 12 × 64)
- This provides good hard negative coverage
- Memory usage remains manageable
- Better learning signal for hierarchy preservation

### Alternative: Balanced Approach

If memory is very constrained:

**Option A: Moderate Both**
- Batch size: 16
- Negatives: 24-32
- Total negatives: 384-512 per batch

**Option B: Prioritize Negatives (Current Direction)**
- Batch size: 12
- Negatives: 32-48
- Total negatives: 384-576 per batch

## Implementation Considerations

### Memory Analysis

**Current (Stage 2):**
- Batch size: 12
- Negatives: 24
- Total embeddings per batch: 12 (anchors) + 12 (positives) + 288 (negatives) = 312

**With More Negatives:**
- Batch size: 12
- Negatives: 48
- Total embeddings per batch: 12 + 12 + 576 = 600

**With More Batches:**
- Batch size: 24
- Negatives: 24
- Total embeddings per batch: 24 + 24 + 576 = 624

**Conclusion**: Similar memory usage, but more negatives provide better learning signal.

### Gradient Quality

**More Negatives:**
- ✅ Better hard negative mining
- ✅ Clearer "not similar" signal
- ✅ Better for fine-grained hierarchy learning
- ✅ More informative gradients per anchor

**More Batches:**
- ✅ More diverse samples per update
- ✅ Lower gradient variance
- ✅ Better coverage of data distribution
- ⚠️ Doesn't directly improve each anchor's learning task

## Final Recommendation

### **For Your Model: Prioritize More Negatives**

**Suggested Configuration:**

```yaml
# Stage 1 (Easy)
n_negatives: 24  # Keep current

# Stage 2 (Medium)
n_negatives: 32-40  # Increase from 24

# Stage 3+ (Hard)
n_negatives: 48-64  # Increase from 28
```

**Why:**
1. **Better hard negative mining**: More negatives = better chance of finding challenging examples
2. **Hierarchy learning**: More negatives help model learn fine-grained distinctions in NAICS tree
3. **Memory efficient**: Uses memory more efficiently than increasing batch size
4. **Aligns with curriculum**: Your stages already increase negatives, suggesting this is the right direction
5. **InfoNCE benefits**: The loss function benefits more from hard negatives than batch diversity

**If Memory Allows:**
- Consider increasing batch size to 16-20 AND keeping negatives at 32-48
- This gives you both benefits: diversity + hard negatives

**If Memory is Tight:**
- Keep batch size at 12
- Prioritize increasing negatives to 32-48
- This is more efficient for InfoNCE loss

## Testing Strategy

1. **Baseline**: Current config (batch=12, negatives=24)
2. **Test 1**: Increase negatives to 32 (batch=12, negatives=32)
3. **Test 2**: Increase batch to 16 (batch=16, negatives=24)
4. **Test 3**: Both (batch=16, negatives=32)

**Monitor:**
- Training loss convergence
- Hierarchy metrics (cophenetic, spearman)
- Memory usage
- Training speed

**Expected Results:**
- More negatives should improve hierarchy metrics
- Larger batches may improve training stability
- Combination may give best results if memory allows

---

*Based on InfoNCE loss mechanics, contrastive learning best practices, and your specific hierarchical embedding use case.*

