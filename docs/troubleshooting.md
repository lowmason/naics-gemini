# NAICS Gemini Troubleshooting Guide

This guide provides diagnostic procedures and solutions for common issues encountered when training hierarchical NAICS embeddings.

## Table of Contents

- [Data Pipeline Issues](#data-pipeline-issues)
- [Training Issues](#training-issues)
- [Model Architecture Issues](#model-architecture-issues)
- [Performance Issues](#performance-issues)
- [Curriculum Issues](#curriculum-issues)
- [Environment Issues](#environment-issues)

---

## Data Pipeline Issues

### Issue: Download Fails from Census Bureau

**Symptoms:**
```
HTTPError: 404 Client Error
Connection timeout after 30 seconds
```

**Diagnosis:**
```bash
# Test direct access
curl -I https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx

# Check if behind firewall
curl -I --proxy http://proxy.company.com:8080 <URL>
```

**Solutions:**

1. **Network timeout:** Increase timeout in `utils/utilities.py`
   ```python
   # Line ~25 in download_with_retry
   timeout: float = 60.0  # Increase from 30.0
   ```

2. **Firewall/proxy:** Configure proxy
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **Manual download:** Download files manually, place in `data/raw/`
   ```bash
   mkdir -p data/raw
   # Download files manually to data/raw/
   # Modify download_data.py to read from local files
   ```

4. **Census.gov down:** Wait and retry, or use cached version
   ```bash
   # If you have old data
   cp backup/naics_descriptions.parquet data/
   ```

### Issue: Excel Parsing Errors

**Symptoms:**
```
ValueError: Sheet 'tbl_2022_title_description_count' not found
polars.exceptions.SchemaError: expected column 'Code' not found
```

**Diagnosis:**
```bash
# Check Excel file structure
python -c "import openpyxl; wb = openpyxl.load_workbook('file.xlsx'); print(wb.sheetnames)"
```

**Solutions:**

1. **Wrong sheet name:** Census Bureau may have updated file
   ```python
   # Update sheet name in download_data.py Config class
   sheet_codes: str = 'NEW_SHEET_NAME'
   ```

2. **Column renamed:** Update schema
   ```python
   # In download_data.py Config
   schema_codes: Dict[str, pl.DataType] = field(
       default_factory=lambda: {
           'NEW_COLUMN_NAME': pl.Utf8,  # Update here
           ...
       }
   )
   ```

3. **Corrupted download:** Clear and re-download
   ```bash
   rm data/raw/*.xlsx
   uv run naics-gemini data preprocess
   ```

### Issue: Missing Descriptions After Processing

**Symptoms:**
```
WARNING: 154 codes at level 4 have empty descriptions
NAICS completed descriptions: Missing (level 4): 154
```

**Diagnosis:**
```bash
# Check which codes are missing
python -c "
import polars as pl
df = pl.read_parquet('data/naics_descriptions.parquet')
missing = df.filter(pl.col('description') == '')
print(missing.select('code', 'level'))
"
```

**Solutions:**

1. **Expected behavior:** Some 4-digit codes rely on child codes
   - Script automatically fills from children
   - Should see: "Filled missing (level 4): 154"

2. **Genuinely missing:** Add manual descriptions
   ```python
   # In download_data.py, add fallback descriptions
   manual_descriptions = {
       '5419': 'Other professional services...',
       ...
   }
   ```

### Issue: Distance Computation Hangs

**Symptoms:**
```
Sector 31: [ 630 nodes, 198,135 pairs]
[hangs indefinitely]
```

**Diagnosis:**
```bash
# Check sector size
python -c "
import polars as pl
df = pl.read_parquet('data/naics_descriptions.parquet')
print(df.group_by('code').agg(pl.len()).sort('len', descending=True))
"
```

**Solutions:**

1. **Large sector:** Sector 31-33 (manufacturing) is huge
   - Expected: 5-10 minutes for sector 31
   - Be patient or optimize algorithm

2. **Memory issue:** Reduce batch size
   ```python
   # In compute_distances.py, process in chunks
   # (Requires code modification)
   ```

3. **CPU bottleneck:** Use multiprocessing
   ```bash
   # Parallel sector processing (requires code changes)
   ```

### Issue: Triplet Generation OOM

**Symptoms:**
```
Killed (process ran out of memory)
MemoryError: Unable to allocate array
```

**Diagnosis:**
```bash
# Check available memory
free -h
# Check triplet count
python -c "
import polars as pl
df = pl.scan_parquet('data/naics_distances.parquet')
print(f'Pairs: {df.select(pl.len()).collect()[0, 0]:,}')
"
```

**Solutions:**

1. **Use lazy evaluation:** Already implemented in Polars
   - Should not load full dataset in memory
   - Verify streaming is working

2. **Reduce precision:** Use smaller dtypes
   ```python
   # In create_triplets.py
   # Change Float32 to Float16 for distances
   ```

3. **Subsample triplets:** For debugging only
   ```python
   # In create_triplets.py, add filter
   triplets = triplets.filter(
       pl.col('anchor_code').cast(pl.Int32) < 550000  # Sample subset
   )
   ```

---

## Training Issues

### Issue: Loss is NaN

**Symptoms:**
```
Epoch 0: train/contrastive_loss = nan
RuntimeWarning: invalid value encountered in acosh
```

**Diagnosis:**
```bash
# Enable anomaly detection
python -c "
import torch
torch.autograd.set_detect_anomaly(True)
# Run training
"
```

**Solutions:**

1. **Hyperbolic projection overflow:** Increase clamping
   ```python
   # In utils/hyperbolic.py, increase epsilon
   norm_v = torch.clamp(norm_v, min=1e-6)  # Was 1e-8
   ```

2. **Lorentzian distance instability:** Tighter clamp
   ```python
   # In model/loss.py
   clamped_dot = torch.clamp(dot_product, max=-1.0 - 1e-4)  # Was 1e-5
   ```

3. **Learning rate too high:** Reduce LR
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     training.learning_rate=1e-5
   ```

4. **Corrupted checkpoint:** Start fresh
   ```bash
   rm -rf checkpoints/01_stage_easy/
   uv run naics-gemini train -c 01_stage_easy
   ```

5. **Mixed precision issue:** Disable bf16
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     training.trainer.precision=32
   ```

### Issue: Loss Not Decreasing

**Symptoms:**
```
Epoch 0: train/contrastive_loss = 4.235
Epoch 1: train/contrastive_loss = 4.198
Epoch 2: train/contrastive_loss = 4.201
...
Epoch 10: train/contrastive_loss = 4.150
```

**Diagnosis:**

1. Check curriculum:
   ```bash
   cat conf/curriculum/01_stage_easy.yaml
   ```

2. Check data loading:
   ```python
   # In Python REPL
   from naics_gemini.data.datamodule import NAICSDataModule
   dm = NAICSDataModule(...)
   dm.setup('fit')
   batch = next(iter(dm.train_dataloader()))
   print(batch.keys())
   print(batch['anchor']['title']['input_ids'].shape)
   ```

3. Monitor gradients:
   ```python
   # Add to training_step in naics_model.py
   for name, param in self.named_parameters():
       if param.grad is not None:
           print(f"{name}: {param.grad.norm()}")
   ```

**Solutions:**

1. **Curriculum too hard:** Start easier
   ```yaml
   # Change from:
   difficulty_buckets: [6, 7, 8]
   # To:
   difficulty_buckets: [1]
   ```

2. **Learning rate too low:** Increase LR
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     training.learning_rate=5e-4
   ```

3. **Insufficient positives:** Relax constraints
   ```yaml
   positive_distance_max: null  # Remove constraint
   ```

4. **Dead ReLU in MoE:** Check activation stats
   ```python
   # Add logging in moe.py forward()
   print(f"Expert outputs: {[e.abs().mean() for e in expert_outputs]}")
   ```

5. **Embedding collapse:** Check embedding variance
   ```python
   # After training
   embeddings = model.encoder(batch)
   print(f"Embedding std: {embeddings.std(dim=0).mean()}")
   # Should be > 0.1; if < 0.01, collapse occurred
   ```

### Issue: Validation Loss Diverging

**Symptoms:**
```
Epoch 0: val/contrastive_loss = 2.100
Epoch 1: val/contrastive_loss = 2.050
Epoch 2: val/contrastive_loss = 2.150
Epoch 3: val/contrastive_loss = 2.300
```

**Diagnosis:**

```python
# Check train vs val loss gap
import pandas as pd
metrics = pd.read_csv('outputs/01_stage_easy/metrics.csv')
print(metrics[['train/loss', 'val/loss']].tail(20))
```

**Solutions:**

1. **Overfitting to curriculum:** Increase diversity
   ```yaml
   # Add more difficulty buckets
   difficulty_buckets: [1, 2, 5]
   bucket_percentages: {1: 0.60, 2: 0.25, 5: 0.15}
   ```

2. **Validation set too small:** Increase val split
   ```yaml
   # In data/default.yaml
   val_split_fraction: 0.05  # Was 0.01
   ```

3. **Different curriculum for val:** Ensure same curriculum
   ```python
   # Check in datamodule.py that val uses same curriculum
   ```

4. **Add regularization:** Increase weight decay
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     training.weight_decay=0.05
   ```

5. **Early stopping:** Enable patience
   ```python
   # Already implemented in training callbacks
   # Adjust patience in conf/training/default.yaml
   ```

### Issue: Training Extremely Slow

**Symptoms:**
```
Epoch 0: 100%|██████| 1000/1000 [02:30:00<00:00,  9.00s/batch]
```

**Diagnosis:**

1. Profile data loading:
   ```python
   import time
   dataloader = dm.train_dataloader()
   
   start = time.time()
   for i, batch in enumerate(dataloader):
       if i >= 10: break
       print(f"Batch {i}: {time.time() - start:.2f}s")
       start = time.time()
   ```

2. Profile forward pass:
   ```python
   import torch
   with torch.profiler.profile() as prof:
       output = model(batch)
   print(prof.key_averages().table())
   ```

**Solutions:**

1. **Data loading bottleneck:** Increase workers
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     data.num_workers=8 \
     data.persistent_workers=true
   ```

2. **Disk I/O slow:** Use SSD, or load to RAM
   ```bash
   # Copy Parquet to tmpfs
   sudo mkdir /mnt/ramdisk
   sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk
   cp data/naics_training_pairs.parquet /mnt/ramdisk/
   ```

3. **CPU bottleneck in negative sampling:** Pre-filter dataset
   ```python
   # Create curriculum-specific Parquet file
   import polars as pl
   df = pl.read_parquet('data/naics_training_pairs.parquet')
   filtered = df.filter(
       # Apply curriculum filters
       pl.col('distance_diff').is_in([7.0])
   )
   filtered.write_parquet('data/stage1_filtered.parquet')
   ```

4. **Model too large:** Disable MoE for debugging
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     model.moe.enabled=false
   ```

---

## Model Architecture Issues

### Issue: MoE Mode Collapse

**Symptoms:**
```
train/load_balancing_loss = 0.450
Expert utilization: [0.95, 0.03, 0.01, 0.01]
```

**Diagnosis:**

```python
# Add logging in moe.py
print(f"Expert counts: {load}")
print(f"Expert importance: {importance}")
```

**Solutions:**

1. **Increase load balancing coefficient:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     model.moe.load_balancing_coef=0.05
   ```

2. **Increase expert diversity:** More experts
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     model.moe.num_experts=8
   ```

3. **Initialize experts differently:**
   ```python
   # In moe.py __init__
   for i, expert in enumerate(self.experts):
       # Different initialization per expert
       torch.nn.init.xavier_uniform_(expert[0].weight, gain=1.0 + 0.1*i)
   ```

4. **Use auxiliary-loss-free balancing:** (Requires implementation)
   ```python
   # See DeepSeek approach (future work)
   ```

### Issue: LoRA Not Learning

**Symptoms:**
```
LoRA weight norms remain constant
Validation loss not improving despite train loss decreasing
```

**Diagnosis:**

```python
# Check LoRA weights are being trained
for name, param in model.named_parameters():
    if 'lora' in name:
        print(f"{name}: requires_grad={param.requires_grad}, norm={param.norm()}")
```

**Solutions:**

1. **LoRA rank too low:** Increase r
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     model.lora.r=16
   ```

2. **LoRA alpha too low:** Increase alpha
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     model.lora.alpha=32
   ```

3. **Wrong target modules:**
   ```python
   # Check in encoder.py that target_modules correct
   # Should be ['query', 'value'] not ['q_proj', 'v_proj']
   ```

4. **Base model frozen:** Verify gradients flow
   ```python
   # LoRA should not freeze base model
   # Check peft config in encoder.py
   ```

### Issue: Hyperbolic Embeddings on Boundary

**Symptoms:**
```
Warning: Many embeddings near boundary of hyperboloid
Embedding norms: [999.8, 1000.1, 999.9, ...]
```

**Diagnosis:**

```python
# Check embedding norms
hyp_emb = model.loss_fn.hyperbolic_proj(euclidean_emb)
norms = torch.sqrt(-hyp_emb[:, 0]**2 + (hyp_emb[:, 1:]**2).sum(dim=1))
print(f"Norms should be ~1/√c: {norms}")
```

**Solutions:**

1. **Curvature too high:** Reduce c
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     loss.curvature=0.1
   ```

2. **Projection learning rate too high:**
   ```python
   # Use separate LR for projection layer
   # (Requires code modification)
   projection_params = model.loss_fn.hyperbolic_proj.parameters()
   optimizer = torch.optim.AdamW([
       {'params': encoder_params, 'lr': 2e-4},
       {'params': projection_params, 'lr': 1e-5}
   ])
   ```

3. **Initialize projection to output small vectors:**
   ```python
   # In HyperbolicProjection.__init__
   torch.nn.init.xavier_uniform_(self.projection.weight, gain=0.01)
   ```

---

## Performance Issues

### Issue: Out of Memory (OOM)

**Symptoms:**
```
CUDA out of memory. Tried to allocate 2.50 GiB
torch.cuda.OutOfMemoryError
Process killed (signal 9)
```

**Diagnosis:**

```python
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Solutions (in order of effectiveness):**

1. **Reduce batch size:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     data.batch_size=16
   ```

2. **Reduce K negatives:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     curriculum.k_negatives=8
   ```

3. **Enable gradient accumulation:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     training.trainer.accumulate_grad_batches=2
   ```

4. **Use lower precision:**
   ```bash
   # If on older GPU without bf16
   uv run naics-gemini train -c 01_stage_easy \
     training.trainer.precision=16-mixed
   ```

5. **Disable MoE temporarily:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     model.moe.enabled=false
   ```

6. **Clear cache between epochs:**
   ```python
   # Add to training_step
   if self.trainer.global_step % 100 == 0:
       torch.cuda.empty_cache()
   ```

### Issue: GPU Underutilized

**Symptoms:**
```
nvidia-smi shows 30% GPU utilization
Training throughput: 0.5 batches/sec (expected 3+)
```

**Diagnosis:**

```bash
# Monitor GPU
nvidia-smi dmon -s ucm

# Check dataloader
# If CPU at 100%, data loading is bottleneck
htop
```

**Solutions:**

1. **Increase batch size:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     data.batch_size=64
   ```

2. **Increase data workers:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     data.num_workers=8
   ```

3. **Pin memory:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     data.pin_memory=true
   ```

4. **Use persistent workers:**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     data.persistent_workers=true
   ```

### Issue: Checkpoint Files Too Large

**Symptoms:**
```
Checkpoint size: 2.5 GB per file
Disk full after 5 epochs
```

**Solutions:**

1. **Save only top K:**
   ```yaml
   # In training callback config
   save_top_k: 1  # Only best model
   ```

2. **Don't save optimizer state:**
   ```python
   # Modify checkpoint callback
   save_weights_only=True
   ```

3. **Compress checkpoints:**
   ```bash
   find checkpoints/ -name "*.ckpt" -exec gzip {} \;
   ```

---

## Curriculum Issues

### Issue: No Training Data After Filtering

**Symptoms:**
```
WARNING: Curriculum filters too restrictive
0 triplets match curriculum constraints
DataLoader: 0 batches
```

**Diagnosis:**

```python
import polars as pl

df = pl.scan_parquet('data/naics_training_pairs.parquet')

# Apply curriculum filters
filtered = df.filter(
    # Your curriculum constraints
    pl.col('positive_distance') <= 2.0
)

print(f"Matching triplets: {filtered.select(pl.len()).collect()[0, 0]:,}")
```

**Solutions:**

1. **Relax positive distance:**
   ```yaml
   positive_distance_max: null  # Or increase to 4.0
   ```

2. **Add more difficulty buckets:**
   ```yaml
   difficulty_buckets: [1, 2, 3, 4, 5]
   ```

3. **Include more levels:**
   ```yaml
   positive_levels: [5, 6]  # Not just [6]
   ```

### Issue: K Negatives Not Available

**Symptoms:**
```
WARNING: Only 4/16 negatives sampled for hardness 7
Falling back to hardness 6
```

**Diagnosis:**

```python
# Count triplets per hardness
import polars as pl
df = pl.read_parquet('data/naics_training_pairs.parquet')

hardness = df.with_columns(
    hardness=pl.when(pl.col('excluded')).then(8)
            .when(pl.col('distance_diff') <= 0.5).then(7)
            .when(pl.col('distance_diff') <= 1.0).then(6)
            # ... etc
)

print(hardness.group_by('hardness').agg(pl.len()))
```

**Solutions:**

1. **Reduce K:**
   ```yaml
   k_negatives: 8  # Instead of 16
   ```

2. **Adjust bucket percentages:**
   ```yaml
   # Reduce allocation to sparse buckets
   bucket_percentages:
     1: 0.70
     6: 0.05  # Was 0.15
   ```

3. **Remove sparse buckets:**
   ```yaml
   # Don't include hardness 7, 8 early on
   difficulty_buckets: [1, 2, 5, 6]
   ```

---

## Environment Issues

### Issue: CUDA Not Available

**Symptoms:**
```
RuntimeError: No CUDA GPUs available
Torch version: 2.5.1+cpu
```

**Diagnosis:**

```bash
uv run python -m naics_gemini.utils.backend
```

**Solutions:**

1. **Install CUDA-enabled PyTorch:**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install matching PyTorch
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Use CPU training (slow):**
   ```bash
   uv run naics-gemini train -c 01_stage_easy \
     training.trainer.accelerator=cpu
   ```

3. **Use Apple Silicon (MPS):**
   ```bash
   # Should auto-detect on M1/M2/M3 Macs
   uv run python -m naics_gemini.utils.backend
   # Should show: MPS (Apple Silicon Metal) available
   ```

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'naics_gemini'
ImportError: cannot import name 'NAICSContrastiveModel'
```

**Solutions:**

1. **Install package:**
   ```bash
   uv sync
   # Or
   pip install -e .
   ```

2. **Check Python path:**
   ```bash
   python -c "import sys; print(sys.path)"
   # Should include your project directory
   ```

3. **Verify installation:**
   ```bash
   uv run python -c "import naics_gemini; print(naics_gemini.__file__)"
   ```

### Issue: Hydra Configuration Errors

**Symptoms:**
```
omegaconf.errors.ConfigAttributeError: Key 'curriculum' not in config
ValueError: Cannot find configuration 'my_curriculum'
```

**Solutions:**

1. **Check config file exists:**
   ```bash
   ls conf/curriculum/my_curriculum.yaml
   ```

2. **Verify YAML syntax:**
   ```bash
   python -c "import yaml; yaml.safe_load(open('conf/curriculum/my_curriculum.yaml'))"
   ```

3. **Clear Hydra cache:**
   ```bash
   rm -rf outputs/.hydra/
   ```

4. **Use absolute path for debugging:**
   ```python
   from hydra import compose, initialize_config_dir
   from pathlib import Path
   
   config_dir = str(Path(__file__).parent / 'conf')
   initialize_config_dir(config_dir=config_dir)
   cfg = compose(config_name='config', overrides=['curriculum=01_stage_easy'])
   ```

---

## General Debugging Workflow

### Step 1: Isolate the Issue

1. **Minimal reproduction:**
   ```bash
   # Test with smallest possible configuration
   uv run naics-gemini train -c 01_stage_easy \
     training.trainer.max_epochs=1 \
     data.batch_size=4 \
     curriculum.k_negatives=4
   ```

2. **Component testing:**
   ```python
   # Test data loading separately
   from naics_gemini.data.datamodule import NAICSDataModule
   dm = NAICSDataModule(...)
   batch = next(iter(dm.train_dataloader()))
   
   # Test model separately
   from naics_gemini.model.naics_model import NAICSContrastiveModel
   model = NAICSContrastiveModel(...)
   output = model(batch['anchor'])
   ```

### Step 2: Enable Verbose Logging

```python
import logging
logging.getLogger('naics_gemini').setLevel(logging.DEBUG)
```

### Step 3: Use Debugger

```bash
# Run with pdb
python -m pdb -m naics_gemini.cli train -c 01_stage_easy

# Or use ipdb
pip install ipdb
python -m ipdb -m naics_gemini.cli train -c 01_stage_easy
```

### Step 4: Check Known Issues

1. GitHub issues: https://github.com/lowmason/naics-gemini/issues
2. Documentation: Read [Architecture](architecture.md) for design details
3. Community: Ask in discussions

---

## Getting Help

If you've exhausted troubleshooting steps:

1. **Gather information:**
   ```bash
   # System info
   uv run python -m naics_gemini.utils.backend
   
   # Config
   cat conf/curriculum/your_curriculum.yaml
   
   # Logs
   tail -n 100 outputs/your_experiment/train.log
   ```

2. **Create minimal reproduction:**
   - Simplest config that demonstrates issue
   - Steps to reproduce
   - Expected vs actual behavior

3. **Open GitHub issue:**
   - Include system info, config, logs
   - Describe troubleshooting steps already tried
   - Link to related issues if found

---

## Prevention Best Practices

✅ **DO:**
- Start with built-in curricula before customizing
- Monitor training curves from the start
- Keep batch sizes conservative initially
- Test on small data subset before full training
- Version control your curriculum configs
- Back up successful checkpoints

❌ **DON'T:**
- Change multiple parameters simultaneously
- Skip validation checks during debugging
- Ignore warnings in console output
- Train without monitoring (use TensorBoard)
- Delete checkpoints without testing first

---

**Quick Reference:**

| Issue | Quick Fix |
|-------|-----------|
| NaN loss | Reduce LR, increase clamping |
| Loss not decreasing | Easier curriculum, check data |
| OOM | Reduce batch_size, k_negatives |
| Slow training | Increase num_workers |
| MoE collapse | Increase load_balancing_coef |
| No training data | Relax curriculum filters |
| Import error | uv sync |

**Next Steps:**
- [Quick Start](quickstart.md) - Setup and training
- [Curriculum Design](curriculum_design_guide.md) - Design effective curricula
- [Architecture](architecture.md) - Understand system design
