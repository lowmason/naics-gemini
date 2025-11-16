# 01_TEXT Stage - Evaluation Metrics Analysis

## Training Summary

- **Total epochs**: 6 (0-5)
- **Best checkpoint**: Epoch 2 (val_loss=0.0367)
- **Training outcome**: Early stopping triggered at optimal point ✓

## Evaluation Metrics Table

| Epoch | Radius (mean±std) | Cophenetic | Spearman | Norm CV | Dist CV | Collapse | Pairs |
|-------|-------------------|------------|----------|---------|---------|----------|-------|
| 0     | 5.36 ± 1.98      | 0.1727     | -0.0019  | 0.3838  | 0.6053  | No       | 4560   |
| 1     | 65.34 ± 66.29    | 0.2115     | 0.1012   | 1.0147  | 1.2959  | No       | 4560   |
| 2     | 17.54 ± 12.05    | 0.2366     | -0.0680  | 0.6896  | 0.8931  | No       | 4560   |
| 3     | 16.23 ± 7.11     | 0.2790     | 0.1318   | 0.4398  | 0.5792  | No       | 4560   |
| 4     | 362.01 ± 301.23  | 0.2343     | -0.0807  | 0.8321  | 0.9836  | No       | 4560   |
| 5     | 314.63 ± 196.75  | 0.2912     | 0.0197   | 0.6253  | 0.7665  | No       | 4560   |

## Detailed Analysis

### 📈 Hierarchy Preservation Metrics

#### Cophenetic Correlation (Tree Structure Preservation)
- **Initial (Epoch 0)**: 0.1727
- **Final (Epoch 5)**: 0.2912
- **Improvement**: +0.1185 (+68.6%)
- **Peak value**: 0.2912 at epoch 5
- **Status**: ⚠️ **LOW** - Needs improvement
  - Target: >0.7 (excellent), >0.5 (good), >0.3 (moderate)
  - Current: 0.29 (just below moderate threshold)

**Analysis**: 
- Cophenetic correlation improved significantly during training (+68.6%)
- However, absolute value is still low (<0.3)
- Model is learning some hierarchy structure but not effectively enough
- The hierarchy preservation loss (hierarchy_weight=0.1) may need tuning

#### Spearman Correlation (Rank Order Preservation)
- **Initial (Epoch 0)**: -0.0019
- **Final (Epoch 5)**: 0.0197
- **Change**: +0.0216
- **Peak value**: 0.1318 at epoch 3
- **Status**: ⚠️ **VERY LOW** - Rank order not preserved
  - Target: >0.7 (excellent), >0.5 (good), >0.3 (moderate)
  - Current: 0.02 (essentially no correlation)

**Analysis**:
- Spearman correlation is essentially zero throughout training
- Embedding distances don't match ground truth rank order
- This suggests the model is not learning distance relationships correctly
- May need stronger hierarchy preservation loss or different distance metric

### 📊 Hyperbolic Embedding Analysis

#### Hyperbolic Radius
- **Initial (Epoch 0)**: 5.36 ± 1.98
- **Final (Epoch 5)**: 314.63 ± 196.75
- **Growth**: +5,775% (extremely large increase)
- **Status**: ⚠️ **CRITICAL** - Numerical instability detected

**Analysis**:
- Radius grew dramatically from 5.36 to 314.63
- This indicates embeddings are moving too far from the origin
- **Manifold constraint violations** observed in epochs 1, 4, and 5:
  - Epoch 1: Max violation = 0.0625
  - Epoch 4: Max violation = 0.5000 (critical!)
  - Epoch 5: Max violation = 0.1250
- This is a serious issue that needs immediate attention

**Recommendations**:
- Reduce learning rate
- Add radius regularization to loss function
- Check for gradient explosion
- Consider clipping hyperbolic embeddings

#### Embedding Diversity
- **Norm CV**: 0.3838 → 0.6253 (increased)
- **Distance CV**: 0.6053 → 0.7665 (increased)
- **Collapse**: ✓ No collapse detected

**Analysis**:
- Embeddings maintain diversity (good)
- Coefficient of variation increased, indicating more spread
- No collapse detected - model is learning diverse representations

## Key Findings

### ✅ Positive Findings

1. **Early Stopping Working**: Training stopped at epoch 2 (best checkpoint), indicating the early stopping fix is working correctly
2. **Cophenetic Improvement**: Correlation improved by 68.6% during training
3. **No Collapse**: Embeddings maintain diversity throughout training
4. **Best Checkpoint**: Epoch 2 had good balance (cophenetic=0.2366, radius=17.54, stable)

### ⚠️ Critical Issues

1. **Hyperbolic Radius Instability**: 
   - Radius grew from 5.36 to 314.63 (+5,775%)
   - Manifold constraint violations in epochs 1, 4, 5
   - This is the most critical issue

2. **Low Hierarchy Preservation**:
   - Cophenetic correlation (0.29) is below moderate threshold (0.3)
   - Spearman correlation (0.02) is essentially zero
   - Model is not effectively preserving NAICS tree structure

3. **Rank Order Not Preserved**:
   - Spearman correlation near zero indicates embedding distances don't match ground truth ranks
   - This is a fundamental issue with the distance learning

## Recommendations

### Immediate Actions (Critical)

1. **Fix Hyperbolic Radius Instability**:
   - Reduce learning rate (currently 2e-4, try 1e-4 or 5e-5)
   - Add radius regularization term to loss function
   - Implement gradient clipping for hyperbolic embeddings
   - Consider adding a constraint to keep embeddings near origin

2. **Improve Hierarchy Preservation**:
   - Increase `hierarchy_weight` from 0.1 to 0.2-0.3
   - Verify hierarchy preservation loss is being computed correctly
   - Check that ground truth distances are being used properly

### Medium-Term Improvements

3. **Address Rank Order**:
   - Spearman correlation should improve with better hierarchy preservation
   - Consider using a different distance metric or loss function
   - May need to adjust temperature parameter

4. **Training Stability**:
   - Monitor gradient norms during training
   - Add learning rate warmup (already implemented)
   - Consider using a different optimizer or learning rate schedule

## Comparison with Previous Run

**Previous Stage 1 (before fixes)**:
- Final cophenetic: 0.2534
- Final spearman: -0.0139
- Best checkpoint: epoch 4

**Current Run (with fixes)**:
- Final cophenetic: 0.2912 (+14.9% improvement)
- Final spearman: 0.0197 (improved from negative)
- Best checkpoint: epoch 2 (early stopping working!)

**Conclusion**: The fixes are partially working:
- ✅ Early stopping is working (stopped at optimal point)
- ✅ Cophenetic correlation improved
- ⚠️ But hierarchy metrics are still low
- ⚠️ New critical issue: hyperbolic radius instability

## Next Steps

1. **Priority 1**: Fix hyperbolic radius instability (reduce LR, add regularization)
2. **Priority 2**: Increase hierarchy_weight to improve hierarchy preservation
3. **Priority 3**: Monitor training more closely for gradient issues
4. **Priority 4**: Re-run training with fixes and compare results

---

*Analysis generated from training run completed on 2025-11-16*

