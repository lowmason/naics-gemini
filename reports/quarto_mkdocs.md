# Hybrid Documentation Strategy: MkDocs + Quarto

**Project:** NAICS Hyperbolic Embedding System\
**Date:** November 2025\
**Author:** Documentation Planning

------------------------------------------------------------------------

## Executive Summary

This report outlines a hybrid documentation approach that preserves your existing MkDocs setup for API reference while introducing Quarto for interactive tutorials and notebooks. This strategy minimizes migration risk, leverages the strengths of both tools, and adds significant value through executable documentation.

------------------------------------------------------------------------

## Current State Assessment

### Existing MkDocs Infrastructure

Your project has a mature documentation setup:

| Component            | Status                                         |
|----------------------|------------------------------------------------|
| MkDocs configuration | `mkdocs.yml` with Material-adjacent theme      |
| API reference pages  | \~15+ modules via mkdocstrings                 |
| Navigation structure | `.nav.yml` with awesome-nav plugin             |
| CI/CD pipeline       | GitHub Actions → GitHub Pages                  |
| Extensions           | pymdownx suite (arithmatex, superfences, etc.) |

### Documentation Gaps

The current setup excels at reference documentation but lacks:

-   Interactive code examples with live outputs
-   Embedding visualizations (Poincaré disk, hyperboloid plots)
-   Training progression demonstrations
-   Reproducible analysis notebooks

------------------------------------------------------------------------

## Hybrid Architecture

```         
naics-embedder/
├── docs/                          # MkDocs (unchanged)
│   ├── index.md
│   ├── overview.md
│   ├── api/
│   │   ├── cli.md
│   │   ├── encoder.md
│   │   └── ...
│   └── .nav.yml
├── mkdocs.yml                     # Existing config
│
├── tutorials/                     # NEW: Quarto notebooks
│   ├── _quarto.yml
│   ├── index.qmd
│   ├── 01-quickstart.qmd
│   ├── 02-understanding-hyperbolic.qmd
│   ├── 03-training-walkthrough.qmd
│   ├── 04-embedding-visualization.qmd
│   └── 05-evaluation-metrics.qmd
│
└── .github/workflows/
    ├── docs.yml                   # MkDocs deployment (existing)
    └── tutorials.yml              # NEW: Quarto deployment
```

### Deployment Strategy

| Site      | URL                                            | Build Tool |
|-----------|------------------------------------------------|------------|
| Main docs | `lowmason.github.io/naics-embedder/`           | MkDocs     |
| Tutorials | `lowmason.github.io/naics-embedder/tutorials/` | Quarto     |

Both sites cross-link to each other via navigation headers.

------------------------------------------------------------------------

## Implementation Plan

### Phase 1: Quarto Setup (Day 1)

**1.1 Install Quarto**

``` bash
# Add to pyproject.toml dev dependencies
# Note: Quarto itself is installed separately, not via pip
uv add --dev jupyter nbformat nbclient
```

Install Quarto CLI from [quarto.org](https://quarto.org/docs/get-started/).

**1.2 Create tutorials directory structure**

``` bash
mkdir -p tutorials
```

**1.3 Create `tutorials/_quarto.yml`**

``` yaml
project:
  type: website
  output-dir: _site

website:
  title: 'NAICS Embedder Tutorials'
  site-url: https://lowmason.github.io/naics-embedder/tutorials/
  repo-url: https://github.com/lowmason/naics-embedder
  repo-actions: [edit, issue]

  navbar:
    background: dark
    left:
      - text: Tutorials
        href: index.qmd
      - text: '← API Docs'
        href: https://lowmason.github.io/naics-embedder/

    right:
      - icon: github
        href: https://github.com/lowmason/naics-embedder

  sidebar:
    style: docked
    search: true
    contents:
      - section: 'Getting Started'
        contents:
          - 01-quickstart.qmd
      - section: 'Concepts'
        contents:
          - 02-understanding-hyperbolic.qmd
      - section: 'Training'
        contents:
          - 03-training-walkthrough.qmd
      - section: 'Analysis'
        contents:
          - 04-embedding-visualization.qmd
          - 05-evaluation-metrics.qmd

format:
  html:
    theme:
      dark: darkly
      light: flatly
    css: styles.css
    toc: true
    toc-depth: 3
    code-fold: false
    code-tools: true
    code-copy: true
    highlight-style: github-dark

execute:
  freeze: auto  # Only re-render when source changes
  cache: true
```

**1.4 Create `tutorials/styles.css`**

``` css
/* Match MkDocs Material aesthetic */
.quarto-title-banner {
  background: linear-gradient(135deg, #009688 0%, #00796b 100%);
}

code {
  font-family: 'Fira Code', monospace;
}
```

### Phase 2: Tutorial Content (Days 2-4)

**2.1 `tutorials/index.qmd`**

```` markdown
---
title: 'NAICS Embedder Tutorials'
subtitle: 'Interactive guides for hyperbolic embedding'
---

Welcome to the interactive tutorial series for the NAICS Hyperbolic Embedding System.

These tutorials complement the [API Reference](https://lowmason.github.io/naics-embedder/)
with executable examples and visualizations.

## Tutorial Series

| Tutorial | Description | Time |
|----------|-------------|------|
| [Quickstart](01-quickstart.qmd) | End-to-end training in 5 minutes | 10 min |
| [Understanding Hyperbolic Space](02-understanding-hyperbolic.qmd) | Visual intuition for Lorentz geometry | 20 min |
| [Training Walkthrough](03-training-walkthrough.qmd) | Deep dive into curriculum learning | 30 min |
| [Embedding Visualization](04-embedding-visualization.qmd) | Poincaré disk and hierarchy plots | 25 min |
| [Evaluation Metrics](05-evaluation-metrics.qmd) | Interpreting hierarchy preservation | 15 min |

## Prerequisites

```{{bash}}
uv sync
uv run naics-embedder data all
```
````

**2.2 `tutorials/02-understanding-hyperbolic.qmd`** (example)

```` markdown
---
title: 'Understanding Hyperbolic Space'
description: 'Visual intuition for the Lorentz model'
execute:
  echo: true
  warning: false
---

## Why Hyperbolic?

Hierarchical data like NAICS has exponential growth at each level.
Euclidean space cannot embed trees without distortion, but hyperbolic
space has room for exponentially many points at each radius.

```{{python}}
#| label: fig-tree-growth
#| fig-cap: 'NAICS hierarchy growth by level'

import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet('data/naics_descriptions.parquet')
level_counts = (
    df.with_columns(pl.col('code').str.len_chars().alias('level'))
    .group_by('level')
    .len()
    .sort('level')
)

plt.figure(figsize=(8, 4))
plt.bar(level_counts['level'], level_counts['len'], color='teal')
plt.xlabel('NAICS Level (digits)')
plt.ylabel('Number of Codes')
plt.title('Exponential Growth of NAICS Hierarchy')
plt.show()
```

## The Lorentz Model

The Lorentz (hyperboloid) model embeds points on the upper sheet of
a two-sheeted hyperboloid in Minkowski space.

```{{python}}
#| label: fig-hyperboloid
#| fig-cap: 'Lorentz hyperboloid visualization'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate hyperboloid surface
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, 2, 30)
U, V = np.meshgrid(u, v)

X = np.sinh(V) * np.cos(U)
Y = np.sinh(V) * np.sin(U)
Z = np.cosh(V)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('x₀ (time)')
ax.set_title('Lorentz Hyperboloid: -x₀² + x₁² + x₂² = -1')
plt.show()
```

## Lorentzian Distance

Unlike Euclidean distance, Lorentzian distance grows logarithmically
with the embedding radius, naturally capturing hierarchy depth.

```{{python}}
#| label: lorentz-distance

import torch
from naics_embedder.text_model.hyperbolic import LorentzDistance

dist_fn = LorentzDistance(curvature=1.0)

# Points at different radii (hierarchy depths)
shallow = torch.tensor([[1.5, 1.0, 0.5]])  # Near origin
deep = torch.tensor([[5.0, 4.0, 2.0]])     # Far from origin

# Normalize to hyperboloid
def to_hyperboloid(x):
    space = x[..., 1:]
    space_sq = (space ** 2).sum(dim=-1, keepdim=True)
    time = torch.sqrt(1 + space_sq)
    return torch.cat([time, space], dim=-1)

shallow_h = to_hyperboloid(shallow)
deep_h = to_hyperboloid(deep)

d = dist_fn(shallow_h, deep_h)
print(f'Lorentzian distance: {d.item():.4f}')
```
````

**2.3 Additional tutorials to create:**

| File | Content Focus |
|----------------------|--------------------------------------------------|
| `01-quickstart.qmd` | Minimal training example, loading pretrained embeddings |
| `03-training-walkthrough.qmd` | SADC curriculum phases, loss curves, checkpointing |
| `04-embedding-visualization.qmd` | Poincaré projection, t-SNE, hierarchy heatmaps |
| `05-evaluation-metrics.qmd` | Interpreting cophenetic correlation, NDCG, collapse detection |

### Phase 3: CI/CD Integration (Day 5)

**3.1 Create `.github/workflows/tutorials.yml`**

``` yaml
name: Deploy Tutorials

on:
  push:
    branches: [main, master]
    paths:
      - 'tutorials/**'
      - '.github/workflows/tutorials.yml'
  workflow_dispatch:

jobs:
  build-deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Setup Quarto
        uses: quarto-dev/quarto-actions/setup@v2

      - name: Prepare data (for notebook execution)
        run: |
          uv run naics-embedder data all
        continue-on-error: true  # Data may be cached

      - name: Render Quarto
        run: |
          cd tutorials
          quarto render

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./tutorials/_site
          destination_dir: tutorials
```

**3.2 Update MkDocs to link to tutorials**

Add to `docs/index.md`:

``` markdown
## Interactive Tutorials

For hands-on learning with executable code, see the
[Tutorial Series](/tutorials/).
```

Add to `mkdocs.yml` navigation:

``` yaml
nav:
  - Home: index.md
  - Tutorials: https://lowmason.github.io/naics-embedder/tutorials/
  # ... rest of nav
```

### Phase 4: Cross-Linking (Day 5)

**4.1 Add "API Docs" link in Quarto navbar** (already in `_quarto.yml` above)

**4.2 Add tutorial callouts in relevant API pages**

Example addition to `docs/api/hyperbolic.md`:

``` markdown
!!! tip "Interactive Tutorial"
    See [Understanding Hyperbolic Space](/tutorials/02-understanding-hyperbolic.html)
    for visual explanations and executable examples.
```

------------------------------------------------------------------------

## Suggested Tutorial Content

### High-Value Tutorials for Your Project

| Priority | Tutorial | Why Valuable |
|----------------------|----------------------|-----------------------------|
| 🔴 High | Embedding Visualization | Users need to see what hyperbolic embeddings look like |
| 🔴 High | Training Walkthrough | SADC curriculum is complex; visual loss curves help |
| 🟡 Medium | Quickstart | Reduces barrier to entry |
| 🟡 Medium | Evaluation Metrics | Helps users interpret training success |
| 🟢 Low | Custom Data | Advanced users bringing their own taxonomies |

### Visualization Ideas

``` python
# Poincaré disk projection (high impact visual)
def lorentz_to_poincare(x):
    '''Project Lorentz point to Poincaré disk.'''
    return x[..., 1:] / (1 + x[..., 0:1])


# Interactive plotly for embedding exploration
import plotly.express as px

embeddings_2d = lorentz_to_poincare(embeddings)
fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    color=naics_levels,
    hover_data={'code': codes, 'title': titles},
    title='NAICS Embeddings in Poincaré Disk'
)
fig.update_layout(
    xaxis=dict(scaleanchor='y', scaleratio=1),
    shapes=[dict(type='circle', x0=-1, y0=-1, x1=1, y1=1)]
)
```

------------------------------------------------------------------------

## Cost-Benefit Analysis

### Benefits of Hybrid Approach

| Benefit                          | Impact                          |
|----------------------------------|---------------------------------|
| Zero disruption to existing docs | No migration risk               |
| Executable examples              | Users learn faster              |
| Visual embedding explanations    | Differentiates from competitors |
| Incremental adoption             | Add tutorials as needed         |
| Separate CI/CD                   | Failures don't block main docs  |

### Costs

| Cost                     | Mitigation                               |
|--------------------------|------------------------------------------|
| Two build systems        | Separate workflows, clear boundaries     |
| Quarto learning curve    | Minimal—similar to Jupyter + Markdown    |
| CI compute for notebooks | `freeze: auto` caches outputs            |
| Cross-site navigation    | Clear "← API Docs" / "Tutorials →" links |

### Effort Estimate

| Phase                          | Effort       |
|--------------------------------|--------------|
| Setup & config                 | 0.5 days     |
| Tutorial content (5 notebooks) | 2-3 days     |
| CI/CD integration              | 0.5 days     |
| Testing & polish               | 1 day        |
| **Total**                      | **4-5 days** |

------------------------------------------------------------------------

## Recommendations

1.  **Start with one high-impact tutorial**: `04-embedding-visualization.qmd` provides immediate value and showcases Quarto's strengths.

2.  **Use `freeze: auto`**: Prevents CI from re-running expensive training code on every push.

3.  **Keep notebooks focused**: Each tutorial should take 15-30 minutes; link to API docs for details.

4.  **Version pin Quarto**: Specify version in CI to avoid breaking changes.

5.  **Consider Quarto for future reports**: The PDF export could generate documentation like your existing `NAICS_Hyperbolic_Embedding_System_Documentation.pdf` from the same source.

------------------------------------------------------------------------

## Appendix: Quick Reference

### Local Development

``` bash
# MkDocs (existing)
uv run mkdocs serve  # http://localhost:8000

# Quarto tutorials
cd tutorials
quarto preview       # http://localhost:4000
```

### Build Commands

``` bash
# MkDocs
uv run mkdocs build  # Output: site/

# Quarto
cd tutorials
quarto render        # Output: tutorials/_site/
```

### File Extensions

| Extension | Tool   | Purpose                            |
|-----------|--------|------------------------------------|
| `.md`     | MkDocs | Static documentation               |
| `.qmd`    | Quarto | Executable notebooks               |
| `.ipynb`  | Either | Quarto can render Jupyter directly |

------------------------------------------------------------------------

*End of Report*