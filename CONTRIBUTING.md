# Contributing to NAICS Gemini

Thank you for your interest in contributing to NAICS Gemini! This guide will help you get started.

## Development Setup

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/lowmason/naics-gemini.git
cd naics-gemini

# Install with dev dependencies
uv sync --all-extras

# Verify installation
uv run python -m naics_gemini.utils.backend
```

### 2. Generate Test Data

```bash
# Generate all datasets
uv run naics-gemini data all
```

---

## Code Style

We use **YAPF for formatting** and **Ruff for linting**.

### Formatting with YAPF

```bash
# Format entire codebase
uv run yapf -ir src/

# Format specific file
uv run yapf -i src/naics_gemini/model/naics_model.py

# Check formatting (dry-run)
uv run yapf -dr src/
```

**Style Guidelines:**
- **Line length**: 100 characters
- **Quotes**: Single quotes (e.g., `'hello'` not `"hello"`)
- **Blank lines**: 2 after top-level functions/classes
- **Method chaining**: Vertical alignment with dot on new line
- **Semantic dividers**: Preserved via high split penalties

**Example:**
```python
# Good: Vertical chaining (YAPF-approved)
result = (
    df
    .filter(pl.col('code').is_not_null())
    .group_by('sector')
    .agg(pl.col('count').sum())
)


# Two blank lines after function
def my_function():
    pass


def another_function():
    pass
```

### Linting with Ruff

```bash
# Check for linting issues
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/
```

**Ruff Rules:**
- **E**: pycodestyle errors
- **F**: Pyflakes
- **I**: isort (import ordering)
- **Q**: flake8-quotes (enforce single quotes)

### Pre-Commit Workflow

**Before committing**, always run:

```bash
# 1. Format code
uv run yapf -ir src/

# 2. Fix linting issues
uv run ruff check --fix src/

# 3. Verify everything passes
uv run yapf -dr src/   # Should output nothing
uv run ruff check src/  # Should output: All checks passed!
```

---

## Type Hints

All public functions must have type hints:

```python
# Good
def process_data(code: str, level: int) -> pl.DataFrame:
    ...

# Bad
def process_data(code, level):
    ...
```

---

## Docstrings

Use **Google-style docstrings**:

```python
def compute_distance(code_i: str, code_j: str, c: float = 1.0) -> float:
    '''
    Compute graph distance between two NAICS codes.
    
    Args:
        code_i: First NAICS code
        code_j: Second NAICS code
        c: Curvature parameter (default: 1.0)
        
    Returns:
        Graph distance between codes
        
    Raises:
        ValueError: If codes don't exist in taxonomy
        
    Example:
        >>> compute_distance("541511", "541512")
        1.0
    '''
    ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=naics_gemini tests/

# Run with verbose output
pytest -v tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Mirror source structure: `src/naics_gemini/model/loss.py` → `tests/test_loss.py`
- Use descriptive test names: `test_lorentz_distance_handles_boundary_case()`

**Example:**
```python
# tests/test_hyperbolic.py
import torch
from naics_gemini.utils.hyperbolic import lorentz_distance

def test_lorentz_distance_same_point():
    '''Distance from point to itself should be zero.'''
    point = torch.randn(1, 769)
    dist = lorentz_distance(point, point)
    assert torch.isclose(dist, torch.tensor(0.0), atol=1e-5)

def test_lorentz_distance_symmetric():
    '''Distance should be symmetric: d(u,v) = d(v,u).'''
    u = torch.randn(1, 769)
    v = torch.randn(1, 769)
    assert torch.isclose(lorentz_distance(u, v), lorentz_distance(v, u))
```

---

## Pull Request Process

### 1. Create a Branch

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Or bugfix branch
git checkout -b fix/issue-123
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation if needed

### 3. Run Pre-Commit Checks

```bash
# Format
uv run yapf -ir src/

# Lint
uv run ruff check --fix src/

# Test
pytest tests/

# Verify
uv run yapf -dr src/
uv run ruff check src/
```

### 4. Commit

```bash
git add .
git commit -m "feat: Add new feature X"

# Commit message format:
# - feat: New feature
# - fix: Bug fix
# - docs: Documentation changes
# - style: Code style changes
# - refactor: Code refactoring
# - test: Test additions/changes
# - chore: Maintenance tasks
```

### 5. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a pull request on GitHub with:
- Clear description of changes
- Link to related issues
- Test results
- Any breaking changes noted

---

## Code Review Checklist

Before requesting review, ensure:

- [ ] Code is formatted with YAPF
- [ ] Code passes Ruff linting
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated if needed
- [ ] Type hints on public functions
- [ ] Docstrings on public functions
- [ ] No debug print statements
- [ ] No commented-out code

---

## Development Tips

### Quick Commands

```bash
# Format and lint in one line
uv run yapf -ir src/ && uv run ruff check --fix src/

# Run tests matching pattern
pytest tests/ -k "test_lorentz"

# Generate test coverage report
pytest --cov=naics_gemini --cov-report=html tests/
open htmlcov/index.html
```

### Working with Data

```bash
# Test with small dataset
uv run naics-gemini data preprocess
# Then modify triplets.py to use .head(1000)

# Validate data pipeline changes
python -c "
import polars as pl
df = pl.read_parquet('data/naics_descriptions.parquet')
assert df.height == 2125
print('✓ Validation passed')
"
```

### Debugging Training

```bash
# Single epoch test
uv run naics-gemini train -c 01_stage_easy \
  training.trainer.max_epochs=1 \
  data.batch_size=4

# Enable debug logging
# Add to train command:
  logging.level=DEBUG
```

---

## Project Structure

```
naics-gemini/
├── src/naics_gemini/
│   ├── data/              # Data loading & streaming
│   ├── data_generation/   # Pipeline scripts
│   ├── model/             # Neural network components
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── conf/                  # Hydra configurations
├── docs/                  # Documentation
└── pyproject.toml         # Dependencies & config
```

---

## Common Issues

### YAPF Not Installed
```bash
uv add --dev yapf
```

### Import Errors
```bash
# Reinstall in editable mode
uv sync
```

### Test Failures
```bash
# Clear cache
pytest --cache-clear tests/

# Run single test with verbose output
pytest -vv tests/test_model.py::test_specific_function
```

---

## Getting Help

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/lowmason/naics-gemini/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/lowmason/naics-gemini/discussions)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to NAICS Gemini!** 🎉
