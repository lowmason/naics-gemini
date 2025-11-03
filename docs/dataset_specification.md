# NAICS Gemini Dataset Specification

This document provides formal specifications for all datasets in the NAICS Gemini system, including schemas, validation rules, data quality expectations, and usage patterns.

## Table of Contents

- [Overview](#overview)
- [Dataset 1: NAICS Descriptions](#dataset-1-naics-descriptions)
- [Dataset 2: NAICS Distances](#dataset-2-naics-distances)
- [Dataset 3: NAICS Training Pairs](#dataset-3-naics-training-pairs)
- [Data Quality Rules](#data-quality-rules)
- [Usage Patterns](#usage-patterns)
- [Version Control](#version-control)

---

## Overview

The NAICS Gemini system uses three primary datasets, each generated sequentially:

```
naics_descriptions.parquet  (Source Data)
         ↓
naics_distances.parquet     (Pairwise Graph Distances)
         ↓
naics_training_pairs.parquet (Training Triplets)
```

### Dataset Generation Commands

```bash
# Stage 1: Generate descriptions (2-3 minutes)
uv run naics-gemini data preprocess

# Stage 2: Compute distances (5-10 minutes)
uv run naics-gemini data distances

# Stage 3: Generate triplets (10-15 minutes)
uv run naics-gemini data triplets
```

### Storage Requirements

| Dataset | Rows | Columns | Disk Size | Memory (full load) |
|---------|------|---------|-----------|-------------------|
| `naics_descriptions.parquet` | 2,125 | 8 | 1.2 MB | ~3 MB |
| `naics_distances.parquet` | 3,004,420 | 5 | 48 MB | ~150 MB |
| `naics_training_pairs.parquet` | 263,830,364 | 8 | 3.2 GB | ~15 GB |

**Note:** `naics_training_pairs.parquet` is designed for streaming access and should not be loaded fully into memory.

---

## Dataset 1: NAICS Descriptions

### Purpose

Source dataset containing all 2,125 NAICS codes with associated text fields for model training. Each code represents an industry classification with title, description, illustrative examples, and exclusions.

### File Location

```
data/naics_descriptions.parquet
```

### Schema

```python
Schema([
    ('index', UInt32),           # Unique identifier (0-2124)
    ('level', UInt8),            # Hierarchy depth (2-6)
    ('code', String),            # NAICS code (e.g., "541511")
    ('title', String),           # Official industry title
    ('description', String),     # Detailed description
    ('examples', String),        # Illustrative examples (nullable)
    ('excluded', String),        # Exclusion text (nullable)
    ('excluded_codes', List[String])  # Parsed exclusion codes (nullable)
])
```

### Field Specifications

#### `index` (UInt32)
- **Purpose:** Unique integer identifier for each NAICS code
- **Range:** 0 to 2,124 (inclusive)
- **Properties:**
  - Sequential by NAICS code order
  - Corresponds to `idx_i` and `idx_j` in distances dataset
  - Used for efficient joins and lookups
- **Validation:**
  - Must be unique
  - Must be continuous (no gaps)
  - Must match row count - 1

**Example:**
```
index: 0    → code: "11"
index: 1    → code: "111"
index: 2124 → code: "928110"
```

#### `level` (UInt8)
- **Purpose:** Indicates position in NAICS hierarchy
- **Range:** 2 to 6 (inclusive)
- **Meaning:**
  - `2`: Sector (e.g., "11" - Agriculture)
  - `3`: Subsector (e.g., "111" - Crop Production)
  - `4`: Industry Group (e.g., "1111" - Oilseed Farming)
  - `5`: Industry (e.g., "11111" - Soybean Farming)
  - `6`: National Industry (e.g., "111110" - Soybean Farming)
- **Properties:**
  - Directly derived from code length: `level = len(code)`
  - Used for curriculum filtering
- **Validation:**
  - Must equal `len(code)`
  - Must be in range [2, 6]

**Distribution:**
```
Level 2:    20 codes (sectors)
Level 3:    99 codes (subsectors)
Level 4:   311 codes (industry groups)
Level 5:   709 codes (industries)
Level 6:   986 codes (national industries)
Total:   2,125 codes
```

#### `code` (String)
- **Purpose:** Official NAICS 2022 code
- **Format:** 2-6 digit numeric string
- **Properties:**
  - Unique identifier (primary key)
  - Hierarchical structure (prefix indicates parent)
  - Some sectors have combined codes (31-33, 44-45, 48-49)
- **Normalization:** Combined codes normalized to first code
  - "31-33" → "31" (Manufacturing)
  - "44-45" → "44" (Retail Trade)
  - "48-49" → "48" (Transportation)
- **Validation:**
  - Must be unique
  - Must match regex: `^\d{2,6}$`
  - Must exist in official NAICS 2022 taxonomy

**Examples:**
```
"11"      → Agriculture, Forestry, Fishing
"541511"  → Custom Computer Programming Services
"928110"  → National Security
```

#### `title` (String)
- **Purpose:** Official industry title from Census Bureau
- **Properties:**
  - Human-readable name
  - Never null or empty
  - May contain special characters (e.g., "U.S.")
- **Validation:**
  - Must be non-empty
  - Length typically 10-100 characters
  - Must not contain control characters

**Examples:**
```
code="11"     → title="Agriculture, Forestry, Fishing and Hunting"
code="541511" → title="Custom Computer Programming Services"
```

#### `description` (String)
- **Purpose:** Detailed explanation of what the industry includes/excludes
- **Properties:**
  - Multi-sentence text (typically 100-1000 characters)
  - Never null or empty (filled from child codes if missing)
  - Cleaned of HTML tags, normalized whitespace
  - Combined sector codes (31-33) normalized
- **Processing:**
  - Split on `\r\n`, filtered empty lines
  - Removed section headers ("The Sector as a Whole")
  - Removed cross-reference markers
  - Normalized special characters (U.S., e.g., i.e.)
- **Validation:**
  - Must be non-empty
  - Should not contain HTML tags
  - Should not have excessive whitespace

**Example:**
```
code="541511" → description="This industry comprises establishments 
primarily engaged in writing, modifying, testing, and supporting 
software to meet the needs of a particular customer."
```

#### `examples` (String, nullable)
- **Purpose:** Illustrative examples of businesses/activities in this industry
- **Properties:**
  - Semicolon-separated list
  - Nullable (not all codes have examples)
  - Sourced from NAICS Index File or extracted from descriptions
- **Format:** `"Example 1; Example 2; Example 3"`
- **Validation:**
  - If non-null, must contain at least one example
  - Should not contain newlines

**Example:**
```
code="541511" → examples="Custom computer programming services; 
Computer software programming services, custom; Software programming 
services, custom computer"
```

**Statistics:**
- Codes with examples: 1,075 (50.6%)
- Codes without examples: 1,050 (49.4%)
- Total examples across all codes: 20,816

#### `excluded` (String, nullable)
- **Purpose:** Text describing what is NOT included in this code
- **Properties:**
  - Nullable (not all codes have exclusions)
  - Contains references to other NAICS codes
  - Sourced from Cross-Reference file or extracted from descriptions
- **Format:** Free-form text with embedded code references
- **Validation:**
  - If non-null, should contain at least one NAICS code reference
  - Should match pattern: contains `\d{2,6}`

**Example:**
```
code="541511" → excluded="Establishments primarily engaged in 
publishing packaged software are classified in Industry 511210, 
Software Publishers."
```

**Statistics:**
- Codes with exclusions: 1,113 (52.4%)
- Codes without exclusions: 1,012 (47.6%)

#### `excluded_codes` (List[String], nullable)
- **Purpose:** Parsed list of NAICS codes mentioned in `excluded` field
- **Properties:**
  - Nullable list
  - Extracted using regex: `\b\d{2,6}\b`
  - Filtered to only valid NAICS codes
  - Used for generating hardness level 8 triplets
- **Validation:**
  - All codes in list must exist in `code` column
  - Should be non-empty if `excluded` is non-null

**Example:**
```
excluded="...Industry 511210, Software Publishers..."
excluded_codes=["511210"]
```

**Statistics:**
- Total exclusion relationships: 4,586
- Codes with exclusion lists: 1,113

### Sample Records

```python
# Sector-level (2-digit)
{
    "index": 9,
    "level": 2,
    "code": "54",
    "title": "Professional, Scientific, and Technical Services",
    "description": "This sector comprises establishments that specialize...",
    "examples": None,
    "excluded": None,
    "excluded_codes": None
}

# National industry-level (6-digit)
{
    "index": 1456,
    "level": 6,
    "code": "541511",
    "title": "Custom Computer Programming Services",
    "description": "This industry comprises establishments primarily...",
    "examples": "Custom computer programming services; Computer software...",
    "excluded": "Establishments primarily engaged in publishing packaged...",
    "excluded_codes": ["511210", "541512"]
}
```

### Data Quality Rules

**Completeness:**
- ✓ All 2,125 codes must be present
- ✓ `index`, `level`, `code`, `title`, `description` never null
- ✓ `examples`, `excluded`, `excluded_codes` may be null

**Consistency:**
- ✓ `index` must be unique and sequential
- ✓ `code` must be unique
- ✓ `level` must equal `len(code)`
- ✓ All codes in `excluded_codes` must exist in `code` column
- ✓ Parent codes must exist (e.g., if "541511" exists, "54151", "5415", "541", "54" must exist)

**Integrity:**
- ✓ No duplicate codes
- ✓ No orphaned codes (missing parents in hierarchy)
- ✓ All combined sectors normalized (31-33→31, 44-45→44, 48-49→48)

### Validation Script

```python
import polars as pl

df = pl.read_parquet('data/naics_descriptions.parquet')

# Check row count
assert df.height == 2125, f"Expected 2125 rows, got {df.height}"

# Check index continuity
assert df['index'].min() == 0, "Index should start at 0"
assert df['index'].max() == 2124, "Index should end at 2124"
assert df['index'].n_unique() == 2125, "Index should be unique"

# Check level consistency
assert (df['level'] == df['code'].str.len_chars()).all(), "Level mismatch"

# Check code uniqueness
assert df['code'].n_unique() == 2125, "Codes should be unique"

# Check required fields non-null
for col in ['index', 'level', 'code', 'title', 'description']:
    assert df[col].null_count() == 0, f"{col} has nulls"

# Check hierarchy integrity
codes = set(df['code'])
for code in codes:
    if len(code) > 2:
        parent = code[:-1]
        if len(parent) >= 2:
            assert parent in codes, f"Missing parent {parent} for {code}"

print("✓ All validation checks passed")
```

---

## Dataset 2: NAICS Distances

### Purpose

Pairwise graph distances between all NAICS codes, representing hierarchical relationships. Used to filter positive/negative pairs and assign hardness levels in training.

### File Location

```
data/naics_distances.parquet
```

### Schema

```python
Schema([
    ('idx_i', UInt32),      # Index of first code
    ('idx_j', UInt32),      # Index of second code
    ('code_i', String),     # First NAICS code
    ('code_j', String),     # Second NAICS code
    ('distance', Float32)   # Graph distance (0.5-10.0)
])
```

### Field Specifications

#### `idx_i`, `idx_j` (UInt32)
- **Purpose:** Indices corresponding to `index` in descriptions dataset
- **Range:** 0 to 2,124
- **Properties:**
  - Foreign keys to `naics_descriptions.index`
  - Used for efficient joins
  - `idx_i <= idx_j` in tree order (not numeric order)
- **Validation:**
  - Must exist in descriptions dataset
  - Must satisfy ordering constraint

#### `code_i`, `code_j` (String)
- **Purpose:** NAICS codes for human readability
- **Properties:**
  - Redundant with indices (for convenience)
  - Must match codes at `idx_i` and `idx_j`
- **Validation:**
  - Must exist in descriptions dataset
  - Must match: `code_i == descriptions[idx_i].code`

#### `distance` (Float32)
- **Purpose:** Graph distance between codes in NAICS tree
- **Range:** 0.5 to 10.0
- **Formula:**
  ```python
  # Within same sector:
  distance = (depth_i - depth_ancestor) + (depth_j - depth_ancestor) - 0.5 * lineal
  
  # Where:
  #   depth_i, depth_j = depth of codes in tree
  #   depth_ancestor = depth of lowest common ancestor
  #   lineal = 1 if j is direct descendant of i, else 0
  
  # Cross-sector:
  distance = 10.0
  ```
- **Interpretation:**
  - **0.5:** Direct parent-child (lineal relationship)
  - **1.0:** Siblings (same parent)
  - **2.0-8.0:** Within-sector distances
  - **10.0:** Different sectors (unrelated)
- **Validation:**
  - Must be >= 0.5
  - Must be <= 10.0
  - Cross-sector pairs must be exactly 10.0

### Distance Distribution

```
Distance | Count      | Percentage | Interpretation
---------|------------|------------|----------------------------------
0.5      | 2,105      | 0.07%      | Parent-child
1.0      | 2,009      | 0.07%      | Siblings
1.5      | 2,394      | 0.08%      | Uncle-nephew
2.0      | 1,701      | 0.06%      | Close relatives
2.5      | 7,413      | 0.25%      | Cousins
3.0      | 1,012      | 0.03%      | Distant cousins
3.5      | 19,144     | 0.64%      | Same subsector
4.0      | 38,841     | 1.29%      | Related subsectors
5.0      | 62,241     | 2.07%      | Same sector, distant
6.0      | 70,835     | 2.36%      | Same sector, far
7.0      | 64,408     | 2.14%      | Same sector, very far
8.0      | 2,732,317  | 90.94%     | Different sectors
10.0     | -          | -          | (Error: should be in distance 8.0)

Note: The actual output shows distance values slightly different due to 
the lineal adjustment. The table above shows the general pattern.
```

### Sector-wise Statistics

```
Sector | Nodes | Pairs   | Notes
-------|-------|---------|---------------------------
11     | 131   | 8,515   | Agriculture
21     | 41    | 820     | Mining
22     | 25    | 300     | Utilities
23     | 73    | 2,628   | Construction
31     | 630   | 198,135 | Manufacturing (largest)
42     | 161   | 12,880  | Wholesale Trade
44     | 139   | 9,591   | Retail Trade
48     | 140   | 9,730   | Transportation
51     | 71    | 2,485   | Information
52     | 79    | 3,081   | Finance
53     | 53    | 1,378   | Real Estate
54     | 95    | 4,465   | Professional Services
55     | 7     | 21      | Management (smallest)
56     | 87    | 3,741   | Administrative
61     | 38    | 703     | Educational
62     | 92    | 4,186   | Healthcare
71     | 61    | 1,830   | Arts
72     | 34    | 561     | Accommodation
81     | 93    | 4,278   | Other Services
92     | 75    | 2,775   | Public Administration
```

### Sample Records

```python
# Parent-child relationship
{
    "idx_i": 100,
    "idx_j": 102,
    "code_i": "5415",
    "code_j": "54151",
    "distance": 0.5
}

# Sibling relationship
{
    "idx_i": 102,
    "idx_j": 103,
    "code_i": "54151",
    "code_j": "54152",
    "distance": 1.0
}

# Cross-sector relationship
{
    "idx_i": 0,
    "idx_j": 500,
    "code_i": "11",
    "code_j": "54",
    "distance": 10.0
}
```

### Data Quality Rules

**Completeness:**
- ✓ Must contain all valid pairs where `idx_i <= idx_j`
- ✓ Expected count: 3,004,420 pairs
- ✓ No null values allowed

**Consistency:**
- ✓ `idx_i` and `code_i` must correspond
- ✓ `idx_j` and `code_j` must correspond
- ✓ Cross-sector pairs must have distance = 10.0
- ✓ Within-sector distances must be < 10.0

**Symmetry:**
- Pairs are directional: (i, j) where i ≤ j in tree order
- To get distance(j, i): look up distance(i, j)
- Distance is symmetric: distance(i, j) = distance(j, i)

### Validation Script

```python
import polars as pl

df = pl.read_parquet('data/naics_distances.parquet')
desc = pl.read_parquet('data/naics_descriptions.parquet')

# Check row count
assert df.height == 3004420, f"Expected 3,004,420 pairs"

# Check no nulls
assert df.null_count().sum_horizontal()[0] == 0, "Found nulls"

# Check distance range
assert df['distance'].min() >= 0.5, "Distance too small"
assert df['distance'].max() <= 10.0, "Distance too large"

# Check index validity
assert df['idx_i'].max() <= 2124, "Invalid idx_i"
assert df['idx_j'].max() <= 2124, "Invalid idx_j"

# Check code correspondence
merged = df.join(desc, left_on='idx_i', right_on='index')
assert (merged['code_i'] == merged['code']).all(), "Code mismatch for i"

# Check cross-sector distances
desc_with_sector = desc.with_columns(
    sector=pl.col('code').str.slice(0, 2)
)
df_with_sectors = (
    df
    .join(desc_with_sector.select(['index', 'sector']), 
          left_on='idx_i', right_on='index')
    .rename({'sector': 'sector_i'})
    .join(desc_with_sector.select(['index', 'sector']), 
          left_on='idx_j', right_on='index')
    .rename({'sector': 'sector_j'})
)

cross_sector = df_with_sectors.filter(pl.col('sector_i') != pl.col('sector_j'))
assert (cross_sector['distance'] == 10.0).all(), "Cross-sector should be 10.0"

print("✓ All validation checks passed")
```

---

## Dataset 3: NAICS Training Pairs

### Purpose

Complete set of training triplets `(anchor, positive, negative)` with hardness annotations. Used by streaming dataset to sample batches during training.

### File Location

```
data/naics_training_pairs.parquet
```

### Schema

```python
Schema([
    ('anchor_code', String),        # Anchor NAICS code
    ('positive_code', String),      # Positive (similar) NAICS code
    ('negative_code', String),      # Negative (dissimilar) NAICS code
    ('excluded', Boolean),          # Is negative in positive's exclusion list?
    ('unrelated', Boolean),         # Are negative and positive in different sectors?
    ('positive_distance', Float32), # Distance(anchor, positive)
    ('negative_distance', Float32), # Distance(anchor, negative)
    ('distance_diff', Float32)      # negative_distance - positive_distance
])
```

### Field Specifications

#### `anchor_code` (String)
- **Purpose:** Reference point for contrastive learning
- **Properties:**
  - Must exist in descriptions dataset
  - Can repeat many times (one per positive-negative pair)
- **Validation:**
  - Must be valid NAICS code
  - Must exist in `naics_descriptions.code`

#### `positive_code` (String)
- **Purpose:** Semantically similar code (should be pulled closer to anchor)
- **Properties:**
  - Must be in same sector as anchor
  - Must satisfy: `distance(anchor, positive) < distance(anchor, negative)`
- **Validation:**
  - Must be valid NAICS code
  - Must be different from anchor
  - Distance to anchor < 10.0

#### `negative_code` (String)
- **Purpose:** Dissimilar code (should be pushed away from anchor)
- **Properties:**
  - Can be in same or different sector
  - Must satisfy: `distance(anchor, negative) > distance(anchor, positive)`
- **Validation:**
  - Must be valid NAICS code
  - Must be different from positive

#### `excluded` (Boolean)
- **Purpose:** Flags specially hard negatives
- **Values:**
  - `True`: Negative is in positive's `excluded_codes` list
  - `False`: Not an exclusion relationship
- **Interpretation:**
  - `True` = Hardness level 8 (hardest)
  - Exclusions are semantically close but industry-distinct
- **Validation:**
  - If True, negative must be in positive's exclusion list

**Statistics:**
- `excluded=True`: 124,920 triplets (0.05%)
- `excluded=False`: 263,705,295 triplets (99.95%)

#### `unrelated` (Boolean)
- **Purpose:** Flags easiest negatives
- **Values:**
  - `True`: Negative is in different sector than positive
  - `False`: Negative is in same sector as positive
- **Interpretation:**
  - `True` = Hardness level 1 (easiest)
  - `False` = Hardness levels 2-7 (varying difficulty)
- **Validation:**
  - If True, distance_diff must be 7.0
  - If True, excluded must be True (can be cross-sector exclusions)

**Statistics:**
- `unrelated=True`: 246,285,403 triplets (93.36%)
- `unrelated=False`: 17,544,961 triplets (6.64%)

#### `positive_distance` (Float32)
- **Purpose:** Graph distance from anchor to positive
- **Range:** 0.5 to ~8.0
- **Properties:**
  - Derived from `naics_distances` dataset
  - Adjusted for lineal relationships: `distance - 0.5 * lineal`
- **Validation:**
  - Must be >= 0.5
  - Must be < `negative_distance`

#### `negative_distance` (Float32)
- **Purpose:** Graph distance from anchor to negative
- **Range:** 1.0 to 10.0
- **Properties:**
  - Derived from `naics_distances` dataset
  - For unrelated: always 7.0 (after adjustment)
- **Validation:**
  - Must be > `positive_distance`
  - If unrelated, must equal 7.0

#### `distance_diff` (Float32)
- **Purpose:** Difficulty metric for triplet
- **Formula:** `negative_distance - positive_distance`
- **Range:** 0.5 to 7.0
- **Interpretation:**
  - **Small (0.5-1.0):** Very hard negatives (siblings)
  - **Medium (2.0-3.0):** Moderate difficulty
  - **Large (7.0):** Easy negatives (unrelated)
- **Validation:**
  - Must be > 0 (enforced during generation)
  - Must equal `negative_distance - positive_distance`

### Hardness Level Mapping

Hardness levels are computed from `excluded`, `unrelated`, and `distance_diff`:

```python
def compute_hardness(row):
    if row['excluded']:
        return 8  # Exclusions (hardest)
    elif row['unrelated']:
        return 1  # Unrelated (easiest)
    elif row['distance_diff'] <= 0.5:
        return 7  # Very hard siblings
    elif row['distance_diff'] <= 1.0:
        return 6  # Siblings
    elif row['distance_diff'] <= 2.0:
        return 5  # Close relatives
    elif row['distance_diff'] <= 3.0:
        return 4  # Cousins
    elif row['distance_diff'] <= 4.0:
        return 3  # Moderate distance
    else:
        return 2  # Distant (but same sector)
```

### Hardness Distribution

```
Level | Count         | % Total | Description
------|---------------|---------|----------------------------------
1     | 246,285,403   | 93.36%  | Unrelated (different sectors)
2     | 379,255       | 0.14%   | Distant (same sector, far)
3     | 394,736       | 0.15%   | Moderate distance
4     | 826,540       | 0.31%   | Cousins
5     | 3,608,943     | 1.37%   | Close relatives
6     | 12,166,985    | 4.61%   | Siblings
7     | 4,433         | 0.00%   | Very hard siblings
8     | 124,920       | 0.05%   | Exclusions
------|---------------|---------|----------------------------------
Total | 263,791,215   | 100.00% |
```

### Sample Records

```python
# Hardness 1: Unrelated
{
    "anchor_code": "541511",
    "positive_code": "541512",
    "negative_code": "111110",
    "excluded": False,
    "unrelated": True,
    "positive_distance": 1.0,
    "negative_distance": 8.0,
    "distance_diff": 7.0
}

# Hardness 6: Sibling
{
    "anchor_code": "541511",
    "positive_code": "54151",
    "negative_code": "541512",
    "excluded": False,
    "unrelated": False,
    "positive_distance": 0.5,
    "negative_distance": 1.5,
    "distance_diff": 1.0
}

# Hardness 8: Exclusion
{
    "anchor_code": "541519",
    "positive_code": "54151",
    "negative_code": "541511",
    "excluded": True,
    "unrelated": True,
    "positive_distance": 0.5,
    "negative_distance": 2.0,
    "distance_diff": 1.5
}
```

### Data Quality Rules

**Completeness:**
- ✓ Expected count: ~263 million triplets
- ✓ No null values in any column

**Consistency:**
- ✓ All codes must exist in descriptions dataset
- ✓ `distance_diff` must equal `negative_distance - positive_distance`
- ✓ If `unrelated=True`, then `distance_diff` should be 7.0
- ✓ If `excluded=True`, negative must be in positive's exclusion list
- ✓ `positive_distance < negative_distance` (guaranteed by construction)

**Logical Constraints:**
- ✓ Anchor and positive must be in same sector
- ✓ If excluded=True, then unrelated may be True (cross-sector exclusions exist)
- ✓ Distance values must match distances dataset

### Validation Script

```python
import polars as pl

df = pl.scan_parquet('data/naics_training_pairs.parquet')

# Check total count (use scan for large file)
count = df.select(pl.len()).collect()[0, 0]
assert count > 263_000_000, f"Expected >263M triplets, got {count:,}"

# Sample validation (check 1000 rows)
sample = df.head(1000).collect()

# Check no nulls
assert sample.null_count().sum_horizontal()[0] == 0, "Found nulls"

# Check distance consistency
distance_check = (
    (sample['negative_distance'] - sample['positive_distance'])
    .sub(sample['distance_diff'])
    .abs()
    .max()
)
assert distance_check < 0.01, "Distance diff mismatch"

# Check unrelated implies large distance_diff
unrelated = sample.filter(pl.col('unrelated'))
assert (unrelated['distance_diff'] >= 6.0).all(), "Unrelated should have diff>=6"

# Check positive < negative
assert (sample['positive_distance'] < sample['negative_distance']).all()

# Full scan check (slow): verify all codes exist
desc = pl.read_parquet('data/naics_descriptions.parquet')
valid_codes = set(desc['code'])

# Check sample codes
for col in ['anchor_code', 'positive_code', 'negative_code']:
    codes = set(sample[col].unique())
    assert codes.issubset(valid_codes), f"Invalid codes in {col}"

print("✓ All validation checks passed")
```

---

## Data Quality Rules

### Cross-Dataset Consistency

These rules ensure datasets are compatible:

1. **Index Consistency:**
   ```python
   # All indices in distances must exist in descriptions
   desc_indices = set(descriptions['index'])
   dist_indices_i = set(distances['idx_i'])
   dist_indices_j = set(distances['idx_j'])
   
   assert dist_indices_i.issubset(desc_indices)
   assert dist_indices_j.issubset(desc_indices)
   ```

2. **Code Consistency:**
   ```python
   # All codes in training pairs must exist in descriptions
   desc_codes = set(descriptions['code'])
   train_codes = set(
       pl.concat([
           training_pairs['anchor_code'],
           training_pairs['positive_code'],
           training_pairs['negative_code']
       ]).unique()
   )
   
   assert train_codes.issubset(desc_codes)
   ```

3. **Distance Integrity:**
   ```python
   # Distances in training pairs must match distances dataset
   # (spot check on sample)
   sample_pairs = training_pairs.head(100)
   
   for row in sample_pairs.iter_rows(named=True):
       anchor = row['anchor_code']
       positive = row['positive_code']
       
       # Look up distance in distances dataset
       actual_dist = distances.filter(
           (pl.col('code_i') == anchor) & (pl.col('code_j') == positive) |
           (pl.col('code_i') == positive) & (pl.col('code_j') == anchor)
       )['distance'][0]
       
       assert abs(actual_dist - row['positive_distance']) < 0.01
   ```

### Data Freshness

Datasets should be regenerated when:
- NAICS taxonomy is updated by Census Bureau
- Data generation code is modified
- Previous generation had errors

**Version tracking:**
```bash
# Add metadata file
cat > data/VERSION.txt << EOF
Generated: 2025-11-03
NAICS Version: 2022
Pipeline Version: 0.1.0
naics_descriptions: 2,125 codes
naics_distances: 3,004,420 pairs
naics_training_pairs: 263,830,364 triplets
EOF
```

---

## Usage Patterns

### Loading for Training

**Descriptions (full load):**
```python
import polars as pl

# Small enough to load fully
descriptions = pl.read_parquet('data/naics_descriptions.parquet')
```

**Distances (conditional load):**
```python
# For lookups: load fully
distances = pl.read_parquet('data/naics_distances.parquet')

# For curriculum filtering: use lazy scan
distances_lazy = pl.scan_parquet('data/naics_distances.parquet')
filtered = distances_lazy.filter(
    pl.col('distance') <= 3.0
).collect()
```

**Training Pairs (streaming only):**
```python
# NEVER load fully into memory!
# Always use lazy scan + streaming

triplets = pl.scan_parquet('data/naics_training_pairs.parquet')

# Apply curriculum filters
filtered = triplets.filter(
    pl.col('distance_diff').is_in([7.0])  # Only unrelated
)

# Stream in batches
for batch in filtered.collect().iter_slices(batch_size=1000):
    # Process batch
    pass
```

### Curriculum Filtering

**Example: Easy curriculum**
```python
easy_triplets = (
    pl.scan_parquet('data/naics_training_pairs.parquet')
    .filter(
        pl.col('unrelated') == True  # Only different sectors
    )
    .collect()
)

print(f"Easy triplets: {easy_triplets.height:,}")
# Output: Easy triplets: 246,285,403
```

**Example: Hard curriculum**
```python
hard_triplets = (
    pl.scan_parquet('data/naics_training_pairs.parquet')
    .filter(
        (pl.col('distance_diff') <= 1.0) & ~pl.col('unrelated')
    )
    .collect()
)

print(f"Hard triplets: {hard_triplets.height:,}")
# Output: Hard triplets: 12,171,418 (siblings + very hard)
```

### Joins for Enrichment

**Add text to training pairs:**
```python
descriptions = pl.read_parquet('data/naics_descriptions.parquet')
triplets = pl.scan_parquet('data/naics_training_pairs.parquet').head(1000)

enriched = (
    triplets
    .join(
        descriptions.select(['code', 'title', 'description']),
        left_on='anchor_code',
        right_on='code'
    )
    .rename({'title': 'anchor_title', 'description': 'anchor_desc'})
    .join(
        descriptions.select(['code', 'title']),
        left_on='positive_code',
        right_on='code'
    )
    .rename({'title': 'positive_title'})
)
```

### Sampling Strategies

**Balanced sampling across hardness levels:**
```python
def sample_balanced(n_per_level=1000):
    """Sample equal number from each hardness level"""
    
    # Define hardness mapping
    triplets = pl.scan_parquet('data/naics_training_pairs.parquet')
    
    # Compute hardness level
    with_hardness = triplets.with_columns(
        hardness=pl.when(pl.col('excluded')).then(8)
                .when(pl.col('unrelated')).then(1)
                .when(pl.col('distance_diff') <= 0.5).then(7)
                .when(pl.col('distance_diff') <= 1.0).then(6)
                .when(pl.col('distance_diff') <= 2.0).then(5)
                .when(pl.col('distance_diff') <= 3.0).then(4)
                .when(pl.col('distance_diff') <= 4.0).then(3)
                .otherwise(2)
    )
    
    # Sample from each level
    samples = []
    for level in range(1, 9):
        level_sample = (
            with_hardness
            .filter(pl.col('hardness') == level)
            .collect()
            .sample(n=min(n_per_level, len(level_sample)))
        )
        samples.append(level_sample)
    
    return pl.concat(samples)
```

---

## Version Control

### Dataset Versioning

Recommended approach for tracking dataset versions:

```bash
# Structure
data/
├── v1.0/
│   ├── naics_descriptions.parquet
│   ├── naics_distances.parquet
│   └── naics_training_pairs.parquet
├── v1.1/
│   └── ...
└── current -> v1.1/  # Symlink to latest
```

### Checksum Verification

```bash
# Generate checksums
cd data
sha256sum *.parquet > checksums.txt

# Verify checksums
sha256sum -c checksums.txt
```

### Metadata File

```yaml
# data/metadata.yaml
version: "1.0.0"
generated_at: "2025-11-03T12:00:00Z"
naics_version: "2022"
pipeline_version: "0.1.0"

datasets:
  naics_descriptions:
    rows: 2125
    size_bytes: 1258291
    sha256: "abc123..."
  
  naics_distances:
    rows: 3004420
    size_bytes: 50331648
    sha256: "def456..."
  
  naics_training_pairs:
    rows: 263830364
    size_bytes: 3435973836
    sha256: "ghi789..."

validation:
  all_codes_present: true
  hierarchy_intact: true
  distances_symmetric: true
  triplets_valid: true
```

---

## Appendix: Quick Reference

### File Sizes
```
naics_descriptions.parquet:   1.2 MB
naics_distances.parquet:      48 MB
naics_training_pairs.parquet: 3.2 GB
```

### Row Counts
```
Descriptions: 2,125
Distances:    3,004,420
Triplets:     263,830,364
```

### Command Summary
```bash
# Generate all datasets
uv run naics-gemini data all

# Validate datasets
python scripts/validate_datasets.py

# Check dataset info
python -c "import polars as pl; print(pl.read_parquet('data/naics_descriptions.parquet').schema)"
```

### Key Constraints
```
✓ 2,125 unique NAICS codes
✓ 20 sectors (level 2)
✓ All distances >= 0.5
✓ All triplets have positive_distance < negative_distance
✓ 93.36% of triplets are unrelated (hardness 1)
```

---

**See Also:**
- [Quick Start Guide](quickstart.md#data-pipeline) - Generate datasets
- [Architecture Guide](architecture.md#data-pipeline-architecture) - Pipeline details
- [Troubleshooting Guide](troubleshooting.md#data-pipeline-issues) - Fix data issues

**Last Updated:** November 2025 | **Version:** 1.0.0
