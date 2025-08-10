# Analog Model Development Changelog

## Project Status
Building an analog-based well production forecaster for DJ Basin (and other basins) using historical well performance data.

---

## Current Work (2025-08-10)

### 🔧 Production Month Indexing Fix
- **Issue Identified:** prod_month field incorrectly indexed in source data
  - 2024 wells showing prod_month 0 = September instead of January
  - Missing first 8 months of production for most 2024 wells
  - Only 51 of 312 2024 wells had correct early months data
- **Root Cause:** ETL pipeline passing through incorrect month indices from source
- **Fix Implemented:** Updated `production_etl_glue.py` to:
  - Calculate prod_month as months since first production (0-indexed)
  - Use Spark's months_between function for accurate calculation
  - Clear existing table data before writing (mode="overwrite" with truncate)
  - Cast to bigint to match database schema
- **Next Steps:**
  1. Deploy updated ETL script to AWS Glue
  2. Re-run production ETL job to repopulate model_prod table
  3. Verify all 2024 wells now have correct month indexing
  4. Re-run early_production.py to rebuild early_rates table

---

## Completed Work (2025-08-10)

### ✅ FULL Candidate Pool Generation Complete
- **Processed all 7,237 target wells** (up from initial 99 test wells)
- **Generated 2,177,201 analog candidate pairs** (up from 16,410)
- **Improved formation match rate to 72.7%** (from 48.9%)
- **301 average candidates per target** (median: 238)
- **Resilient processing:** Implemented resume capability after connection timeout
- **Optimized batch size:** Reduced from 50 to 25 to prevent timeouts
- **Cleanup:** Removed 7 obsolete scripts and old ETL files

### ✅ Step 3: Curve Embeddings Complete
- **Generated PCA embeddings for 7,272 wells**
- **4 components explain 92.3% of oil production variance**
- **Created `src/features/embeddings.py` module**
- **Stored embeddings in `data.curve_embeddings` table**
- **Saved PCA models and scalers to `models/embeddings/`**
- **Key insights:** Months 4-7 are most important for characterizing production curves

### ✅ Step 4: Baseline Picker Complete
- **Implemented simple distance-based analog selector**
- **Criteria:** 2018+ vintage, ±20% lateral length, same formation, <15 miles, top 20 by distance
- **Created `src/models/baseline.py` module**
- **Evaluation results:** 21.5% average cumulative error on test set
- **P50 error: 17%**, P90 error: 34% on 2020+ vintage wells
- **Average 19.5 analogs per well, 2.7 miles average distance**

### ✅ Step 5: Feature Engineering Complete
- **Created `src/features/feature_builder.py` module**
- **Engineered 34 features** for ML training:
  - Distance and spatial features
  - Lateral length ratios and deltas
  - Design parameter ratios (proppant, fluid)
  - Formation and operator matching
  - Vintage gaps and seasonality
  - Production intensity metrics
- **Prepared training data:** 49,083 analog pairs from 500 wells
- **Label strategy:** Revenue-weighted production error scoring
- **10.6% of pairs are "good" analogs** (< 20% error)

### ✅ Step 6: LightGBM LambdaRank Model Complete
- **Removed incorrect XGBoost implementations** (3 files deleted)
- **Created `src/models/lightgbm_ranker.py`** - Learning-to-rank approach
- **Training approach:** 
  - Rank candidates within each target's pool
  - Labels based on revenue-weighted MAE after warping
  - Graded labels (0-3) based on error percentiles
- **Performance:** 20.7% average error (improvement over 21.5% baseline)
- **Top features:** distance, fluid/proppant ratios, lateral length
- **Key insight:** Ranking formulation better than regression for this problem

### ✅ Infrastructure Improvements
- Added command-line argument parsing to `candidate_pool.py`
- Created `resume_candidate_pool.py` for fault-tolerant processing
- Successfully handled database connection issues with retry logic

## Completed Work (2025-08-09)

### ✅ ETL Pipeline Fixed & Data Loaded
- **Fixed** Spark 3.0 compatibility issues with Parquet timestamp handling
- **Resolved** column name mismatches (state_well_id vs well_api)
- **Added** explicit type casting for PostgreSQL compatibility
- **Optimized** joins using leftsemi for better performance
- **Successfully loaded:**
  - 13,545 Denver Basin wells into `model_wells`
  - 919,870 production records into `model_prod`

### ✅ Model Implementation (Steps 1-2) - FULLY OPERATIONAL
- **Step 1:** Early Production Table Builder (`src/data/early_production.py`)
  - Processed **7,272 wells** with 9-month production data
  - Average 9-month cumulative: 64,800 bbls oil
  - Fixed numpy array serialization to PostgreSQL
  - Batch processing with progress tracking
  
- **Step 2:** Candidate Pool Generator (`src/data/candidate_pool.py`)
  - ~~Generated **16,410 analog pairs** from 99 target wells~~ [See 2025-08-10 update]
  - ~~Average 166 candidates per target (median: 82)~~
  - ~~Average distance between analogs: 8.4 miles~~
  - ~~48.9% formation match rate~~

### ✅ Performance Optimizations
- **Added spatial indices** on geometry columns (GIST indices)
- **Added B-tree indices** on filtering columns (dates, lateral_length, etc.)
- **Implemented batch inserts** using psycopg2.extras.execute_batch
- **Result:** Candidate queries run in ~0.5 seconds (down from timeouts)

### ✅ Infrastructure Scripts
- `add_spatial_indices.py` - Adds all necessary database indices
- `test_spatial.py` - Tests spatial query performance
- `monitor_pipeline.py` - Real-time pipeline progress monitoring
- `check_data.py` - Verifies data import success

---

## Current Status - ML MODEL COMPLETE ✅

Steps 1-6 of the analog model are **FULLY OPERATIONAL**:
- Data successfully imported from Energy Domain (13,545 wells, 919,870 production records)
- Early production arrays built for 7,272 wells
- Analog candidates generated for ALL wells (2.17M pairs)
- Curve embeddings created with PCA (92.3% variance explained)
- Baseline model: 21.5% average error
- LightGBM LambdaRank model: **20.7% average error** (4% improvement)
- Feature engineering complete with 22 features
- All performance issues resolved
- Ready for inference pipeline and deployment (Steps 7-8)

---

## Next Steps (In Order)

### ~~1. Expand Candidate Pool Generation~~ ✅ COMPLETE
### ~~2. Step 3: Curve Embeddings~~ ✅ COMPLETE
### ~~3. Step 4: Baseline Picker~~ ✅ COMPLETE
### ~~4. Step 5: Feature Engineering~~ ✅ COMPLETE
### ~~5. Step 6: LightGBM LambdaRank Model~~ ✅ COMPLETE

### 6. Complete Remaining Steps
- **Step 7:** Inference Pipeline - Production deployment
- **Step 8:** Test Harness - Evaluation framework
- **Step 9:** (Optional) Meta-corrector for residuals

---

## File Structure
```
dde-model/
├── config/database.yaml        # Database configuration
├── src/
│   ├── data/                   # Data pipeline (Steps 1-2 COMPLETE ✅)
│   │   ├── early_production.py # Step 1: Build 9-month arrays
│   │   ├── candidate_pool.py   # Step 2: Generate analog candidates
│   │   └── db_connector.py     # Database connection manager
│   ├── features/               # Feature engineering (Steps 3,5 COMPLETE ✅)
│   │   ├── embeddings.py       # Step 3: PCA curve embeddings
│   │   └── feature_builder.py  # Step 5: ML feature engineering
│   └── models/                 # Model implementations (Steps 4,6 COMPLETE ✅)
│       ├── baseline.py         # Step 4: Distance-based baseline
│       └── lightgbm_ranker.py  # Step 6: LightGBM LambdaRank model
├── models/                     # Saved model artifacts
│   ├── embeddings/            # PCA models and scalers
│   └── lightgbm_ranker.pkl    # Trained LightGBM model
├── data/ml_features/          # Prepared training data
├── sql/                       # Database scripts & indexes
├── etl_scripts/               # AWS Glue ETL jobs
└── *.py                       # Various utility scripts
```

---

## Key Metrics (As of 2025-08-10)

| Metric | Value |
|--------|-------|
| Wells with production data | 13,545 |
| Wells with 9-month arrays | 7,272 |
| Production records | 919,870 |
| Analog candidate pairs | **2,177,201** ✅ |
| Unique target wells | **7,237** ✅ |
| Avg candidates per target | **301** |
| Median candidates per target | **238** |
| Query performance | ~0.5 seconds |
| Formation match rate | **72.7%** |
| Same operator rate | **49.4%** |
| Avg distance between analogs | **8.7 miles** |
| Avg vintage gap | **2.5 years** |

---

## Technical Decisions

1. **Filtered Data Approach:** Only import DJ Basin horizontal wells (2010+) instead of billions of rows
2. **Separate Tables:** `model_*` tables optimized for modeling vs. general `ed_*` tables
3. **Spatial Indexing:** PostGIS GIST indices for sub-second distance queries
4. **Batch Processing:** execute_batch for 100x faster inserts
5. **No Data Leakage:** Temporal constraints prevent future data in training

---

## Dependencies
- PostgreSQL with PostGIS (spatial queries)
- AWS Glue 4.0 (ETL pipeline)
- Python 3.8+ with psycopg2, pandas, numpy
- scikit-learn (PCA, preprocessing)
- XGBoost (analog scoring model)

---

## Commands

```bash
# Run full pipeline
python src/data/early_production.py              # Build 9-month arrays
python src/data/candidate_pool.py                # Generate candidates for all wells
python src/data/candidate_pool.py --limit 100    # Generate candidates for 100 wells (testing)

# Resume if interrupted
python resume_candidate_pool.py                  # Fault-tolerant resume from last checkpoint

# Add indices for performance
python add_spatial_indices.py

# Check data status
python check_data.py

# Monitor progress
python monitor_pipeline.py
```