# Developer Quick Reference

**NAICS Gemini Development Cheat Sheet**

---

## 🚀 One-Time Setup

```bash
# Clone and install
git clone https://github.com/lowmason/naics-gemini.git
cd naics-gemini
uv sync --all-extras

# Generate test data
make data
# Or: uv run naics-gemini data all

# Install pre-commit hook (optional)
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

## 💻 Daily Development Workflow

### Using Make (Easiest)

```bash
# Format and fix
make fix

# Check everything
make check

# Run tests
make test
```

### Manual Commands

```bash
# 1. Format code (YAPF)
uv run yapf -ir src/

# 2. Lint and fix (Ruff)
uv run ruff check --fix src/

# 3. Run tests
uv run pytest tests/
```

---

## 📋 All Make Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all commands |
| `make install` | Install dependencies |
| `make format` | Format with YAPF |
| `make lint` | Lint with Ruff |
| `make check` | Check format & lint |
| `make fix` | Format + auto-fix |
| `make test` | Run tests |
| `make test-cov` | Tests with coverage |
| `make data` | Generate datasets |
| `make train-easy` | Train easy curriculum |
| `make clean` | Remove temp files |

---

## 🎨 YAPF Formatting

### Basic Commands
```bash
# Format everything
uv run yapf -ir src/

# Format one file
uv run yapf -i src/naics_gemini/model/loss.py

# Check without modifying
uv run yapf -dr src/

# See diff
uv run yapf -d src/naics_gemini/model/loss.py
```

### Style Guide
- **Line length**: 100 chars
- **Quotes**: Single (`'` not `"`)
- **Blank lines**: 2 after functions/classes
- **Chaining**: Vertical with dot-first

**Good:**
```python
result = (
    df
    .filter(pl.col('code').is_not_null())
    .group_by('sector')
    .agg(pl.col('count').sum())
)


def my_function():
    pass


def another_function():
    pass
```

---

## 🔍 Ruff Linting

### Basic Commands
```bash
# Check all
uv run ruff check src/

# Auto-fix
uv run ruff check --fix src/

# Check one file
uv run ruff check src/naics_gemini/model/loss.py

# Show rule explanation
uv run ruff rule F401
```

### Common Issues

| Error | Meaning | Fix |
|-------|---------|-----|
| F401 | Unused import | Remove or use import |
| E501 | Line too long | Reformat with YAPF |
| I001 | Import order | Run `ruff check --fix` |
| Q000 | Double quotes | Change to single quotes |

---

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_hyperbolic.py

# Specific test
pytest tests/test_hyperbolic.py::test_lorentz_distance

# With coverage
pytest --cov=naics_gemini tests/

# Verbose
pytest -vv tests/

# Stop on first failure
pytest -x tests/

# Match pattern
pytest -k "lorentz" tests/
```

### Write Tests
```python
# tests/test_example.py
import torch
from naics_gemini.utils.hyperbolic import lorentz_distance

def test_distance_symmetric():
    '''Distance should be symmetric.'''
    u = torch.randn(1, 769)
    v = torch.randn(1, 769)
    assert torch.isclose(
        lorentz_distance(u, v), 
        lorentz_distance(v, u)
    )
```

---

## 📦 Pre-Commit Checklist

Before committing:

```bash
# 1. Format
make format

# 2. Fix linting
make lint

# 3. Test
make test

# 4. Check everything passed
make check
```

Or use the automated hook:
```bash
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

## 🐛 Debugging

### Common Issues

**"yapf: command not found"**
```bash
uv add --dev yapf
```

**"Import errors"**
```bash
uv sync
```

**"Tests fail"**
```bash
pytest --cache-clear tests/
pytest -vv tests/  # Verbose output
```

**"YAPF and Ruff disagree"**
```bash
# Run YAPF first, then Ruff
uv run yapf -ir src/
uv run ruff check --fix src/
```

---

## 📝 Git Workflow

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes and format/lint
make fix
make test

# Commit (hook will run automatically if installed)
git add .
git commit -m "feat: Add new feature"

# Push
git push origin feature/my-feature
```

---

## 🎯 Training Quick Start

```bash
# Generate data (once)
make data

# Train easy curriculum
make train-easy

# Monitor
tensorboard --logdir outputs/01_stage_easy

# Train with overrides
uv run naics-gemini train -c 01_stage_easy \
  training.trainer.max_epochs=10 \
  data.batch_size=64
```

---

## 📊 Data Commands

```bash
# Generate all datasets
make data

# Individual stages
uv run naics-gemini data preprocess
uv run naics-gemini data distances
uv run naics-gemini data triplets

# Clean generated data
make data-clean
```

---

## 🔧 Configuration

Edit `pyproject.toml` for:
- YAPF style settings (line length, chaining, etc.)
- Ruff linting rules (imports, quotes, etc.)
- Dependencies

Edit `conf/` for:
- Training curricula
- Model architecture
- Data loading
- Loss functions

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide |
| [docs/quickstart.md](docs/quickstart.md) | Getting started |
| [docs/curriculum_design_guide.md](docs/curriculum_design_guide.md) | Training strategy |
| [docs/architecture.md](docs/architecture.md) | System design |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Debug guide |

---

## 💡 Pro Tips

**One-line format & lint:**
```bash
uv run yapf -ir src/ && uv run ruff check --fix src/
```

**Watch for changes:**
```bash
# Install watchdog
uv add --dev watchdog

# Auto-format on save (example)
watchmedo shell-command --patterns="*.py" --recursive \
  --command='uv run yapf -i "${watch_src_path}"' src/
```

**Quick validation:**
```bash
python -c "
import polars as pl
df = pl.read_parquet('data/naics_descriptions.parquet')
assert df.height == 2125
print('✓ Data valid')
"
```

---

## 🆘 Getting Help

- 📖 Full docs: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/lowmason/naics-gemini/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/lowmason/naics-gemini/discussions)

---

**Keep this reference handy!** 📌
