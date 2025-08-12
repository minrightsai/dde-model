# Multi-Basin Analog Production Forecasting System

## System Overview

### What It Does
This system predicts oil & gas production for new wells by finding and analyzing similar historical wells ("analogs"). Originally built for the DJ Basin, it now supports multiple basins through a configurable architecture. It uses a combination of spatial analysis, production curve characterization, and machine learning to forecast the first 9 months of production.

### Complete Pipeline: From Raw Data to Model Comparison

#### Step 1: Data Ingestion (AWS Glue ETL)
```
S3 Parquet Files → AWS Glue → PostgreSQL
```
1. **Wells ETL** (`etl_scripts/wells_etl_glue.py`): 
   - Reads well metadata from S3 parquet files
   - Filters by basin (DJ/Bakken) using lat/lon boundaries
   - Loads ~23,961 wells into `model_wells` table (~3 minutes)

2. **Production ETL** (`etl_scripts/production_etl_glue.py`):
   - Reads monthly production from S3 parquet files  
   - Joins to `model_wells` to filter by basin wells only
   - Loads ~1.5M records into `model_prod` table (~18 minutes)
   - Updates `first_prod_date` using MIN(prod_date)

#### Step 2: Data Preparation
3. **Early Production Arrays** (`src/data/early_production.py`):
   - Identifies peak production month (0-6 window)
   - Aligns all wells to peak month
   - Creates 9-month production arrays
   - Result: 10,078 wells in `early_rates` table

#### Step 3: Feature Engineering
4. **Analog Candidates** (`src/data/candidate_pool.py`):
   - Finds wells within 20 miles of each target
   - Pre-computes distance, formation match, operator match
   - Result: 2.47M candidate pairs in `analog_candidates` table

5. **Curve Embeddings** (`src/features/embeddings.py`):
   - PCA on production curves (4 components)
   - Captures decline curve shapes
   - Result: 10,078 embeddings in `curve_embeddings` table

6. **Feature Builder** (`src/features/feature_builder.py`):
   - Creates 34 features (spatial, geological, completion)
   - Includes ratios and interactions for ML

#### Step 4: Model Training & Evaluation
7. **Baseline Model** (`src/models/baseline.py`):
   - Simple distance-weighted averaging
   - Selects top 20 closest analogs
   - Applies production warping (lateral length, proppant)

8. **LightGBM Model** (`src/models/lightgbm_ranker.py`):
   - Machine learning ranking model
   - Uses all 34 features to score analogs
   - Selects top 20 by ML score

#### Step 5: Model Comparison
9. **Evaluation** (H1 2024 test set - 304 wells):
   - Weighted MAE metric (early months weighted higher)
   - **Baseline: 2,018 bbls/month median error**
   - **LightGBM: 2,460 bbls/month median error**
   - **Result: Baseline outperforms ML by 22%**

### How It Works

#### Data Pipeline
```
Energy Domain S3 → AWS Glue ETL → PostgreSQL → Model Pipeline → Predictions
```

**Data Flow:**
- **Source:** Oil/gas production data from Energy Domain (S3 parquet files)
- **ETL:** AWS Glue job (`production_etl_glue.py`) processes and loads to PostgreSQL
- **Database:** PostgreSQL with PostGIS for spatial queries
- **Processing:** Python pipeline generates features and predictions

#### 2. Core Components

**Data Tables (PostgreSQL):**
- `model_wells`: 13,545 DJ Basin wells with metadata
- `model_prod`: ~1M monthly production records  
- `early_rates`: 10,078 wells with 9-month production arrays (peak-aligned)
- `analog_candidates`: 2.47M pre-computed well pairs with similarity metrics
- `curve_embeddings`: 10,078 PCA embeddings of production curves

**Pipeline Steps:**
1. **Early Production Arrays** (`src/data/early_production.py`)
   - Builds 9-month production arrays from peak month
   - Normalizes and log-transforms for modeling
   
2. **Candidate Generation** (`src/data/candidate_pool.py`)
   - Finds potential analogs within 20 miles
   - Pre-computes distance, formation match, operator match
   - Average 320 candidates per target well

3. **Curve Embeddings** (`src/features/embeddings.py`)
   - PCA dimensionality reduction on production curves
   - 4 components explain 62% oil, 66% gas variance
   - Captures decline curve shape characteristics

4. **Feature Engineering** (`src/features/feature_builder.py`)
   - 34 features including spatial, geological, completion design
   - Ratios and interactions for ML model

5. **Models:**
   - **Baseline** (`src/models/baseline.py`): Distance-based selection
   - **LightGBM** (`src/models/lightgbm_ranker.py`): ML ranking model

#### 3. Prediction Process

For a new well:
1. Query analog_candidates table for pre-computed candidates
2. Score/rank candidates using model (baseline or LightGBM)
3. Select top 20 analogs
4. Apply production warping based on lateral length and proppant
5. Weight by distance (baseline) or equally (LightGBM)
6. Average to get 9-month production forecast

### Current Performance

**Test Set:** 304 H1 2024 wells with actual production data

**Weighted MAE (barrels/month, weighted toward early months):**
| Model | Median | Mean | P10 | P90 |
|-------|--------|------|-----|-----|
| Baseline | **2,018** | 2,600 | 673 | 5,169 |
| LightGBM | 2,460 | 3,186 | 1,012 | 6,100 |

**Key Insight:** Simple distance-based baseline outperforms complex ML model by 22%

### File Structure
```
dde-model/
├── etl_scripts/               # AWS Glue ETL jobs
│   └── production_etl_glue.py
├── src/
│   ├── config/                # Basin configuration
│   │   └── basin_config.py   # Multi-basin parameters
│   ├── data/                  # Data pipeline
│   │   ├── early_production.py
│   │   ├── candidate_pool.py
│   │   └── db_connector.py
│   ├── features/              # Feature engineering
│   │   ├── embeddings.py
│   │   └── feature_builder.py
│   ├── models/                # Model implementations
│   │   ├── baseline.py
│   │   └── lightgbm_ranker.py
│   └── evaluation/            # Evaluation metrics
│       └── metrics.py
├── models/                    # Saved model artifacts
│   ├── embeddings/           # PCA models
│   └── lightgbm_ranker.pkl  # Trained LightGBM
├── evaluate_baseline.py      # Baseline evaluation script
└── evaluate_lightgbm.py      # LightGBM evaluation script
```

---

## Recent Changes & Improvements

### 2025-08-11 - Multi-Basin Configuration System
- **NEW:** Created `BasinConfig` class for basin-specific parameters
  - Supports DJ Basin, Bakken, Eagle Ford configurations
  - Configurable distance limits (15-25 miles per basin)
  - Basin-specific lateral tolerances, vintage years, formations
  - Warping coefficients tuned per basin geology
- **UPDATED:** Core modules to use basin configuration
  - `baseline.py`: Now accepts basin parameter, uses config for all parameters
  - `candidate_pool.py`: Basin-specific filtering with API prefixes
  - `early_production.py`: Configurable analysis windows per basin
  - ETL pipeline: Added `TARGET_BASINS` parameter for multi-basin support
- **FIXED:** Critical bugs discovered during implementation
  - Coordinate fallback bug: Wells no longer default to (0,0) when missing lat/lon
  - SQL parameter mismatch: Fixed mixing of f-strings and placeholders
  - pandas.read_sql compatibility: Switched to cursor.execute for complex queries
  - Escape character issue: Fixed LIKE clauses with proper %% escaping
  - Formation case sensitivity: Database uses UPPERCASE formations
- **ADDED:** Migration scripts and test utilities
  - `sql/add_basin_columns.sql`: Adds basin_name to all tables
  - `test_basin_config.py`: Comprehensive test suite
  - `quick_test_basin.py`: Quick validation script
- **RESULT:** System ready for multi-basin deployment
  - DJ Basin functionality unchanged (backward compatible)
  - Successfully tested with 2024 wells: 20 analogs found, predictions generated

### 2025-08-11 - Evaluation Metric Overhaul
- **NEW:** Implemented Weighted Mean Absolute Error (MAE) metric
  - Weights: [3, 3, 2, 2, 1.5, 1.5, 1, 1, 1] for months 1-9
  - Early months weighted more heavily (economic importance)
  - Shared metric function in `src/evaluation/metrics.py`
- **REMOVED:** Old cumulative percentage error metric
- **RESULT:** More interpretable errors in barrels, not percentages

### 2025-08-11 - Test Set Expansion & Model Re-evaluation
- **Fixed:** Evaluation scripts limiting to 100 wells arbitrarily
- **Expanded:** Test set from 86 → 304 wells (3.5x increase)
- **Regenerated:** Curve embeddings for all 10,078 wells
- **Finding:** Baseline (2,018 bbls MAE) beats LightGBM (2,460 bbls MAE)

### 2025-08-11 - Data Pipeline Fix
- **Issue:** early_rates table missing 90% of 2024 wells
- **Root Cause:** Using unreliable `model_wells.first_prod_date`
- **Fix:** Modified `early_production.py` to use actual MIN(prod_date)
- **Impact:** 
  - Before: 79 wells from 2024
  - After: 540 wells from 2024 (6.8x increase)
  - Analog candidates regenerated: 2.47M pairs

### 2025-08-10 - Peak Month Alignment
- **Problem:** 91.5% of wells showed artificial ramp-up
- **Solution:** Implemented peak month detection and alignment
- **Method:** Window functions identify peak in months 0-6, reset arrays
- **Result:** Average month 0 production: 18,049 bbls (realistic)

### 2025-08-10 - Production Month Indexing Fix
- **Issue:** prod_month field incorrectly indexed (2024 wells starting in September)
- **Fix:** ETL pipeline calculates months_between(prod_date, first_prod)
- **Impact:** All 2024 wells now correctly indexed from January

### 2025-08-09 - Initial Pipeline Implementation
- **ETL:** Fixed Spark 3.0 compatibility, loaded 13,545 wells
- **Spatial Indexing:** Added PostGIS GIST indices for fast queries
- **Batch Processing:** 100x faster inserts with execute_batch
- **Infrastructure:** Added monitoring and resume capabilities

---

## Next Steps

### Immediate Priorities
1. **Run basin migration** - Execute `python run_basin_migration.py` to add basin_name columns
2. **Deploy baseline model** to production (best performer)
3. **Retrain LightGBM** with weighted MAE objective
4. **Expand coverage** to remaining ~1,200 wells without candidates

### Model Improvements
- Ensemble baseline + LightGBM predictions
- Add decline curve features
- Implement cross-validation
- Handle outlier predictions (P90 cases)

### System Enhancements
- API endpoints for real-time predictions
- ~~Extend to other basins beyond DJ~~ ✅ **COMPLETED** - Multi-basin support implemented
- Add gas production focus
- Extend beyond 9-month horizon

---

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| **Basins Supported** | **3 (DJ, Bakken, Eagle Ford)** |
| Total wells in system | 13,545 |
| Wells with 9-month arrays | 10,078 |
| Production records | ~1M |
| Analog candidate pairs | 2.47M |
| Wells with candidates | 7,734 |
| Average candidates/well | 320 |
| H1 2024 test wells | 304 |
| **Baseline Weighted MAE** | **2,018 bbls** |
| **LightGBM Weighted MAE** | **2,460 bbls** |

---

## Technical Stack
- **Data:** PostgreSQL 13+ with PostGIS
- **ETL:** AWS Glue 4.0 (PySpark)
- **ML:** Python 3.8+, scikit-learn, LightGBM
- **Key Libraries:** psycopg2, pandas, numpy
- **Infrastructure:** AWS S3, AWS Glue, PostgreSQL RDS

---

## [2025-01-12] - Wells-First ETL Pipeline Implementation
### Added
- Complete redesign of ETL pipeline with wells-first approach for massive performance gains
- py4j integration for executing UPDATE statements directly in AWS Glue
- Geographic filtering using precise lat/lon boundaries for DJ Basin and Bakken
- Consistent basin naming convention: "dj"/"bakken" for basin, "dj_basin"/"bakken" for basin_name
- Production-based first_prod_date calculation using actual MIN(prod_date) from production data
- Peak month alignment logic (0-6 month window) for consistent production curve analysis

### Changed
- **BREAKING**: Switched from production-first to wells-first ETL execution order
- ETL pipeline now requires Wells ETL to run before Production ETL
- Eliminated dependency on data.wells table - direct S3 to model_wells loading
- Reduced quality filters to essential only (lat/lon not null) for better data retention
- Updated job configurations to use clean script names (removed _new versions)

### Performance Improvements  
- **Wells ETL**: Reduced from 30-45 minutes to ~3 minutes (90% improvement)
- **Production ETL**: Reduced from 60-90 minutes to ~18 minutes (75% improvement)  
- **Total Pipeline**: Reduced from 90+ minutes to ~20 minutes (78% improvement)
- Massive data reduction: Filter production data to basin wells only (23,961 wells vs millions)

### Fixed
- Resolved circular dependency between Wells ETL and Production ETL
- Fixed inconsistent basin naming across tables
- Eliminated secondary quality filters that were removing all valid wells
- Resolved UPDATE statement execution in AWS Glue using py4j workaround
- Fixed column mapping issues (sub_basin, primary_fluid, formation fields)

### Technical Details
- Wells loaded: 23,961 total wells (DJ: 9,454, Bakken: 11,156 with production)
- Production records: 1,531,769 total records (DJ: 690,673, Bakken: 841,096)
- First production dates updated: 20,610 wells via py4j UPDATE statements
- Geographic filtering: CO (39.51-42.19°N, -105.13 to -101.99°W), ND/MT (46.66-48.99°N, -105.26 to -102.00°W)
- Date filtering: spud_date >= 2014-01-01, production_date >= 2014-01-01