# ETL Pipeline Implementation - Wells-First Approach

This directory contains the complete implementation of the new ETL pipeline as documented in `ETL_FIX.md`.

## Files Overview

### Main ETL Scripts
- `wells_etl_glue_new.py` - Wells ETL (Job 1): Loads well headers with geographic and date filtering
- `production_etl_glue_new.py` - Production ETL (Job 2): Processes production data for basin wells
- `update_first_prod_dates.sql` - Post-ETL update script for first_prod_date correction

### Job Configurations
- `wells-job-config.json` - AWS Glue job configuration for Wells ETL
- `production-job-config-new.json` - AWS Glue job configuration for Production ETL

### Pipeline Management
- `run_etl_pipeline.sh` - Complete pipeline execution script with monitoring
- `validate_pipeline.sql` - Data quality validation queries

### Legacy Files (for reference)
- `wells_etl_glue.py` - Original Wells ETL (replaced by wells_etl_glue_new.py)
- `production_etl_glue.py` - Original Production ETL (replaced by production_etl_glue_new.py)
- `production-job-config.json` - Original job config

## Execution Order

### 1. Setup
```bash
# Upload new scripts to S3
aws s3 cp wells_etl_glue_new.py s3://aws-glue-assets-277707132474-us-east-1/scripts/
aws s3 cp production_etl_glue_new.py s3://aws-glue-assets-277707132474-us-east-1/scripts/
```

### 2. Run Pipeline
```bash
# Option A: Use the automated script
./run_etl_pipeline.sh

# Option B: Manual execution
aws glue start-job-run --job-name wells-etl-job
# Wait for completion...
aws glue start-job-run --job-name production-etl-job
# Wait for completion...
psql -h minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com -d dde -U minrights -f update_first_prod_dates.sql
```

### 3. Validation
```bash
psql -h minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com -d dde -U minrights -f validate_pipeline.sql
```

## Key Improvements

1. **Wells-First Approach**: Massive data reduction by filtering production data to basin wells only
2. **Geographic Filtering**: Precise basin assignment based on lat/lon coordinates
3. **Date Filtering**: Early filtering to wells spudded >= 2014-01-01
4. **Consistent Basin Naming**: Standardized "dj"/"bakken" and "dj_basin"/"bakken" conventions
5. **Production-Based Dates**: first_prod_date derived from actual production MIN(date)

## Expected Performance

- **Wells ETL**: 30-45 minutes (processes ~millions of well headers)
- **Production ETL**: 60-90 minutes (much faster due to basin pre-filtering)
- **Update Script**: <5 minutes

## Data Flow

```
Wells ETL (Job 1):
S3 Well Headers → Geographic Filter (CO/ND/MT) → Date Filter (>=2014) → model_wells

Production ETL (Job 2):  
S3 Production → Join with model_wells → Peak Alignment → model_prod

Update Script:
model_prod → Calculate first_prod_date → Update model_wells
```

## Validation Checks

The pipeline includes comprehensive validation:
- Basin consistency between tables
- Geographic boundary compliance  
- Production history requirements (9+ months)
- Date accuracy and alignment
- Data volume and quality metrics

## Next Steps After Pipeline

Once the ETL pipeline completes successfully, run:

```bash
python -m src.data.early_production --basin dj_basin
python -m src.data.early_production --basin bakken
python -m src.data.candidate_pool --basin dj_basin
python -m src.data.candidate_pool --basin bakken
python -m src.features.embeddings
```