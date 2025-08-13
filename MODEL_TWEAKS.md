# Model Improvements & Tweaks

## 1. Data Quality Filter - Remove Wells with Sparse Production

### Problem
Wells with >2 zero production months are currently included in the pipeline but:
- Don't get embeddings (filled with zeros)
- Make poor analogs due to unreliable production patterns
- Add noise to model training

### Current State
```
early_rates (16,342 wells) → includes wells with spotty production
    ↓
candidate_pool → uses all wells as potential analogs
    ↓
curve_embeddings (10,078 wells) → excludes wells with >2 zero months
    ↓
Model training → fills missing embeddings with [0,0,0,0]
```

### Solution
Remove wells with >2 zero months at the source in `early_production.py`:

```python
# Update the initial wells query in early_production.py (line 104-131)
wells_query = f"""
WITH wells_with_production AS (
    SELECT 
        p.well_id,
        MIN(p.prod_date) as actual_first_prod_date,
        COUNT(DISTINCT p.prod_month) as months_available,
        -- Add quality check for zero months
        SUM(CASE WHEN p.oil_mo_prod = 0 OR p.oil_mo_prod IS NULL THEN 1 ELSE 0 END) as zero_months
    FROM {self.prod_table} p
    GROUP BY p.well_id
    HAVING COUNT(DISTINCT p.prod_month) >= {self.months_to_analyze}
    AND SUM(CASE WHEN p.oil_mo_prod = 0 OR p.oil_mo_prod IS NULL THEN 1 ELSE 0 END) <= 2  -- MAX 2 ZERO MONTHS
)
...
"""
```

### Implementation Steps
1. Update `early_production.py` with the zero month filter
2. Rebuild `data.early_rates` table: `python -m src.data.early_production --basin dj && python -m src.data.early_production --basin bakken`
3. Regenerate candidate pools: `python -m src.data.candidate_pool_fast --basin dj_basin && python -m src.data.candidate_pool_fast --basin bakken`
4. Regenerate embeddings: `python -m src.features.embeddings`
5. Retrain models: `python -m src.models.lightgbm_ranker --basins dj_basin bakken --parallel`

### Expected Impact
- ~20-30% fewer wells, but all high quality
- 100% of wells will have meaningful embeddings
- Better model performance due to cleaner training data
- More reliable analog selection

### Validation Query
```sql
-- Check impact before implementing
SELECT 
    basin_name,
    COUNT(*) as total_wells,
    SUM(CASE WHEN zero_months_count > 2 THEN 1 ELSE 0 END) as wells_to_remove,
    ROUND(100.0 * SUM(CASE WHEN zero_months_count > 2 THEN 1 ELSE 0 END) / COUNT(*), 1) as percent_removed
FROM data.early_rates
GROUP BY basin_name;
```

---

## 2. Replace PCA Embeddings with Peak Production Features

### Problem
Current PCA embeddings have temporal data leakage:
- Embeddings use full 9-month production history
- When predicting for new wells, candidate embeddings include "future" information
- Can't create embeddings for new wells (no 9-month history yet)
- Model performance artificially inflated in backtesting

### Current State
```
curve_embeddings table:
- oil_embedding: [2.34, -0.89, 0.45, -0.12]  # Abstract PCA components
- Created using FULL 9-month curves
- Missing for ~40% of wells (filled with zeros)
```

### Solution
Replace abstract embeddings with interpretable peak production features:

```python
# New peak_features.py (replaces embeddings.py)
def create_peak_features(self) -> pd.DataFrame:
    """Create peak production features instead of embeddings"""
    
    query = """
    SELECT 
        well_id,
        basin_name,
        lateral_length,
        proppant_per_ft,
        oil_m1_9,
        gas_m1_9,
        -- Calculate peaks
        (SELECT MAX(unnest) FROM unnest(oil_m1_9)) as peak_oil,
        (SELECT MAX(unnest) FROM unnest(gas_m1_9)) as peak_gas,
        -- Month when peak occurred (1-indexed)
        array_position(oil_m1_9, (SELECT MAX(unnest) FROM unnest(oil_m1_9))) as peak_oil_month,
        array_position(gas_m1_9, (SELECT MAX(unnest) FROM unnest(gas_m1_9))) as peak_gas_month
    FROM data.early_rates
    """
    
    df = pd.read_sql(query, conn)
    
    # Normalized peaks (per 1000 ft lateral)
    df['peak_oil_per_kft'] = df['peak_oil'] / (df['lateral_length'] / 1000)
    df['peak_gas_per_kft'] = df['peak_gas'] / (df['lateral_length'] / 1000)
    
    # Peak efficiency (peak per proppant)
    df['peak_oil_per_ppf'] = df['peak_oil'] / (df['proppant_per_ft'] + 1)
    
    # Decline rate from peak to month 9
    df['oil_decline_rate'] = (df['peak_oil'] - df['oil_m1_9'].apply(lambda x: x[8])) / (df['peak_oil'] + 1)
    df['gas_decline_rate'] = (df['peak_gas'] - df['gas_m1_9'].apply(lambda x: x[8])) / (df['peak_gas'] + 1)
    
    return df
```

### Update ranking_features.py
```python
# Replace embedding features with peak features
# Remove: 'embedding_pc1', 'embedding_pc2', 'embedding_pc3', 'embedding_pc4'
# Add these instead:

features['candidate_peak_oil'] = candidates_df['peak_oil']
features['candidate_peak_gas'] = candidates_df['peak_gas']
features['peak_oil_ratio'] = candidates_df['peak_oil'] / (target_well['peak_oil'] + 1)
features['peak_gas_ratio'] = candidates_df['peak_gas'] / (target_well['peak_gas'] + 1)
features['log_peak_oil_ratio'] = np.log(features['peak_oil_ratio'].clip(lower=0.1))
features['peak_month_diff'] = candidates_df['peak_oil_month'] - target_well['peak_oil_month']
features['early_peak'] = (candidates_df['peak_oil_month'] <= 3).astype(int)
features['decline_rate_diff'] = candidates_df['oil_decline_rate'] - target_well['oil_decline_rate']
```

### New Database Table
```sql
-- Replace data.curve_embeddings with data.peak_features
CREATE TABLE data.peak_features (
    well_id TEXT PRIMARY KEY,
    basin_name TEXT,
    peak_oil NUMERIC,
    peak_gas NUMERIC,
    peak_oil_month INTEGER,
    peak_gas_month INTEGER,
    peak_oil_per_kft NUMERIC,
    peak_gas_per_kft NUMERIC,
    peak_oil_per_ppf NUMERIC,
    oil_decline_rate NUMERIC,
    gas_decline_rate NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_peak_features_basin ON data.peak_features(basin_name);
CREATE INDEX idx_peak_features_peak_oil ON data.peak_features(peak_oil);
```

### Implementation Steps
1. Create new `peak_features.py` module (based on embeddings.py)
2. Run to create `data.peak_features` table
3. Update `ranking_features.py` to use peak features instead of embeddings
4. Update `lightgbm_ranker.py` to join with peak_features instead of curve_embeddings
5. Retrain models with new features

### Expected Impact
- **No data leakage** - Peak typically reached in months 1-3
- **Better interpretability** - "High peak wells" vs abstract PCA components
- **100% coverage** - All wells have peak values (no missing embeddings)
- **Simpler** - 8 clear features vs 4 abstract components
- **Works for new wells** - Can calculate peak after just 3 months

---

## 3. Refined Feature Set with Time-Aware Well Spacing

### Problem
Current features don't capture well spacing and field development state at the time each candidate well started producing. This is critical because:
- A well that was "isolated" in 2018 might be surrounded by 2024
- Field development intensity affects production
- We need to know what the field looked like when the candidate started, not what it looks like now

### Final Feature Set (17 features)

#### Distance (2)
- `distance_mi` - Distance between target and candidate wells
- `log_distance` - Log-transformed distance

#### Lateral Length (3)
- `length_delta` - Difference in feet (candidate - target)
- `log_length_ratio` - Log of (candidate_length / target_length)
- `abs_log_length_ratio` - Magnitude of length difference

#### Completion Design (2)
- `ppf_ratio` - Proppant ratio (target_ppf / candidate_ppf)
- `log_ppf_ratio` - Log-transformed proppant ratio

#### Operator (1)
- `same_operator` - Binary: same operator or not

#### Temporal (1)
- `vintage_gap_years` - Years between completions (target - candidate)

#### Peak Production (4)
- `candidate_peak_oil` - Candidate's peak monthly oil production
- `peak_oil_ratio` - candidate_peak / target_peak
- `log_peak_oil_ratio` - Log-transformed peak ratio
- `decline_rate_diff` - Difference in decline rates from peak to month 9

#### **NEW: Time-Aware Well Spacing (4)**
These MUST be calculated as of the candidate well's first production date:
- `wells_within_1mi_at_start` - Count of wells within 1 mile when candidate started
- `wells_within_3mi_at_start` - Count of wells within 3 miles when candidate started  
- `distance_to_nearest_at_start` - Distance to closest neighbor when candidate started
- `distance_to_second_nearest_at_start` - Distance to 2nd closest when candidate started

### Critical Implementation Detail: Temporal Filtering

```sql
-- WRONG - uses current field state
SELECT COUNT(*) as wells_within_1mi
FROM early_rates
WHERE ST_DWithin(candidate.geom, other.geom, 1609.34)

-- CORRECT - uses field state when candidate started
SELECT COUNT(*) as wells_within_1mi_at_start
FROM early_rates
WHERE ST_DWithin(candidate.geom, other.geom, 1609.34)
AND other.first_prod_date <= candidate.first_prod_date  -- CRITICAL: Only count wells that existed when candidate started
```

### Why Time-Aware Spacing Matters

**Example: Candidate Well from 2018**
- In 2018 (when it started): 2 wells within 1 mile → relatively isolated
- In 2024 (now): 15 wells within 1 mile → crowded field
- **Using current count would be wrong** - the well produced its 9 months when field was empty!

### Implementation Approach

1. **Pre-compute spacing features** during candidate pool generation
2. **Store in analog_candidates table** with other pre-computed features
3. **Always filter by time** - only count wells that existed before candidate's first production

```sql
-- Add columns to analog_candidates table
ALTER TABLE data.analog_candidates ADD COLUMN wells_within_1mi_at_start INTEGER;
ALTER TABLE data.analog_candidates ADD COLUMN wells_within_3mi_at_start INTEGER;
ALTER TABLE data.analog_candidates ADD COLUMN distance_to_nearest_at_start NUMERIC;
ALTER TABLE data.analog_candidates ADD COLUMN distance_to_second_nearest_at_start NUMERIC;
```

### Expected Impact
- **Accurate analogs** - Compares wells in similar development contexts
- **No temporal leakage** - Only uses information available at the time
- **Better predictions** - Accounts for depletion and interference effects properly