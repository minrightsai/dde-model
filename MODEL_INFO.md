# Multi-Basin Analog Production Forecasting System
## Model Information & Pipeline Guide

Last Updated: 2025-08-12

---

## 🚀 Quick Start - Complete Pipeline

### Prerequisites
- AWS credentials configured for Glue access
- PostgreSQL connection (host: minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com)
- Python 3.8+ with dependencies installed

### Full Pipeline Execution Order

```bash
# Step 1: AWS Glue ETL (Run in AWS Console or CLI)
# Run these jobs IN ORDER - Wells must complete before Production
aws glue start-job-run --job-name wells-etl-job        # ~3 minutes
# Wait for completion, then:
aws glue start-job-run --job-name production-etl-job   # ~18 minutes

# Step 2: Local Data Preparation
python -m src.data.early_production --basin dj         # <1 minute
python -m src.data.early_production --basin bakken     # <1 minute

# Step 3: Generate Analog Candidates (spatial relationships)
python -m src.data.candidate_pool_fast --basin dj_basin    # ~1 minute
python -m src.data.candidate_pool_fast --basin bakken      # ~3 minutes

# Step 4: Create Curve Embeddings
python -m src.features.embeddings                      # ~2 minutes

# Step 5: Train Models (parallel for both basins)
python -m src.models.lightgbm_ranker --basins dj_basin bakken --parallel  # ~5 minutes

# Step 6: Evaluate Models
python -m src.evaluation.model_evaluation --basin dj_basin --year 2024   # ~1 minute
python -m src.evaluation.model_evaluation --basin bakken --year 2024     # ~10 minutes
```

**Total Pipeline Time: ~45 minutes**

---

## 📊 Current Model Performance (2024 H1 Out-of-Sample)

### DJ Basin (74 test wells)
| Model | Median MAE | Mean MAE | P10 | P90 | Status |
|-------|------------|----------|-----|-----|--------|
| **LightGBM** ✓ | **2,371** | 3,547 | 992 | 8,054 | **WINNER (-2.8%)** |
| Baseline | 2,438 | 3,079 | 1,064 | 5,926 | |

### Bakken (429 test wells)
| Model | Median MAE | Mean MAE | P10 | P90 | Status |
|-------|------------|----------|-----|-----|--------|
| **Baseline** ✓ | **5,285** | 6,705 | 2,085 | 12,682 | **WINNER (-2.6%)** |
| LightGBM | 5,428 | 6,543 | 2,580 | 11,636 | |

**Metric**: Weighted Mean Absolute Error (bbls/month)
- Weights: [3, 3, 2, 2, 1.5, 1.5, 1, 1, 1] for months 1-9
- Early months weighted more heavily due to economic importance

---

## 🏗️ Pipeline Architecture

### AWS Glue Components (Cloud)

#### 1. Wells ETL (`etl_scripts/wells_etl_glue.py`)
- **Source**: S3 parquet files (s3://spe-dj-basin/parquet-files/)
- **Target**: PostgreSQL `data.model_wells` table
- **Records**: ~24,000 wells (DJ: 11,052, Bakken: 12,915)
- **Runtime**: ~3 minutes
- **Key Operations**:
  - Geographic filtering (lat/lon boundaries)
  - Basin assignment ('dj' or 'bakken')
  - Formation mapping (reservoir → formation)

#### 2. Production ETL (`etl_scripts/production_etl_glue.py`)
- **Source**: S3 parquet files
- **Target**: PostgreSQL `data.model_prod` table
- **Records**: ~1.5M production records
- **Runtime**: ~18 minutes
- **Key Operations**:
  - Filters to basin wells only (massive data reduction)
  - Calculates prod_month index
  - Updates first_prod_date using MIN(prod_date)

### Local Components (Python)

#### Data Preparation
- **early_production.py**: Creates 9-month production arrays aligned to peak month
- **candidate_pool_fast.py**: Spatial join to find analogs within 15-20 miles
- **embeddings.py**: PCA dimensionality reduction on production curves

#### Models
- **baseline.py**: Simple distance-weighted averaging
- **lightgbm_ranker.py**: Machine learning ranking model with basin support

#### Evaluation
- **model_evaluation.py**: Compares model performance on test data

---

## 🗄️ Database Schema

### Core Tables
| Table | Records | Description |
|-------|---------|-------------|
| `data.model_wells` | 23,967 | Well metadata with basin, formation, completion |
| `data.model_prod` | 1.5M | Monthly production data |
| `data.early_rates` | 16,342 | 9-month production arrays (peak-aligned) |
| `data.analog_candidates` | 4.6M | Pre-computed spatial relationships |
| `data.curve_embeddings` | 10,078 | PCA features of production curves |

### Basin Distribution
- **DJ Basin**: 5,514 wells with candidates (1.3M pairs)
- **Bakken**: 6,084 wells with candidates (3.3M pairs)

---

## 🎯 Model Features

### Baseline Model
- Top 20 closest wells within same formation
- Distance-weighted averaging
- Production warping based on lateral length and proppant

### LightGBM Model
**Top 10 Features by Importance:**

**DJ Basin:**
1. distance_x_operator (1087.70)
2. length_x_ppf (196.84)
3. distance_mi (145.76)
4. vintage_gap_years (137.92)
5. length_delta (90.13)

**Bakken:**
1. distance_mi (24259.23)
2. vintage_gap_years (3320.97)
3. distance_x_operator (2796.17)
4. same_operator (2288.56)
5. log_ppf_ratio (1362.93)

---

## 🔧 Configuration

### Basin Parameters (`src/config/basin_config.py`)
```python
DJ_BASIN:
  - Distance limit: 15 miles
  - Lateral tolerance: ±20%
  - Min vintage: 2014

BAKKEN:
  - Distance limit: 20 miles
  - Lateral tolerance: ±25%
  - Min vintage: 2014
```

### Database Connection (`config/database.yaml`)
```yaml
host: minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com
database: dde
user: minrights
password: [configured]
```

---

## 📁 File Structure

```
dde-model/
├── etl_scripts/               # AWS Glue ETL jobs
│   ├── wells_etl_glue.py
│   └── production_etl_glue.py
├── src/
│   ├── config/               # Configuration
│   │   └── basin_config.py
│   ├── data/                 # Data pipeline
│   │   ├── early_production.py
│   │   ├── candidate_pool_fast.py
│   │   └── db_connector.py
│   ├── features/             # Feature engineering
│   │   ├── embeddings.py
│   │   ├── feature_builder.py
│   │   └── ranking_features.py
│   ├── models/               # Model implementations
│   │   ├── baseline.py
│   │   └── lightgbm_ranker.py
│   └── evaluation/           # Model evaluation
│       ├── metrics.py
│       └── model_evaluation.py
├── models/                   # Saved model artifacts
│   ├── lightgbm_ranker_dj_basin.pkl
│   └── lightgbm_ranker_bakken.pkl
└── config/
    └── database.yaml
```

---

## 🚨 Important Notes

1. **ETL Order**: Wells ETL MUST complete before Production ETL (wells-first approach)
2. **Basin Names**: 
   - Basin field: 'dj' or 'bakken'
   - Basin_name field: 'dj_basin' or 'bakken'
3. **First Production Date**: Calculated from MIN(prod_date), not stored in wells table
4. **Formation Data**: 75% DJ wells and 94% Bakken wells have formation data
5. **Memory Requirements**: Bakken training requires ~2GB RAM (2.8M candidate pairs)

---

## 📈 Performance Insights

### Why Different Models Win
- **DJ Basin (LightGBM wins)**: Complex geology benefits from ML feature interactions
- **Bakken (Baseline wins)**: Simpler, more homogeneous geology where distance matters most

### Deployment Recommendation
Use a hybrid approach:
- DJ Basin predictions → LightGBM model
- Bakken predictions → Baseline model

---

## 🔄 Updating Models

To retrain models with new data:

```bash
# 1. Run ETL to update database
aws glue start-job-run --job-name wells-etl-job
aws glue start-job-run --job-name production-etl-job

# 2. Regenerate features
python -m src.data.early_production --basin dj
python -m src.data.early_production --basin bakken
python -m src.data.candidate_pool_fast --basin dj_basin
python -m src.data.candidate_pool_fast --basin bakken

# 3. Retrain models
python -m src.models.lightgbm_ranker --basins dj_basin bakken --parallel

# 4. Evaluate performance
python -m src.evaluation.model_evaluation --basin dj_basin --year 2024
python -m src.evaluation.model_evaluation --basin bakken --year 2024
```

---

## 📞 Support

For issues or questions about the pipeline:
- Check `CHANGELOG.md` for recent changes
- Review error logs in AWS Glue console
- Database connection issues: Verify PostgreSQL credentials
- Model performance: Check data quality and completeness