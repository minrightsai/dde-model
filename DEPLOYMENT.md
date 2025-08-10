# Deployment Instructions for Model Data Pipeline

## Overview
Instead of working with massive unfiltered tables, we'll create focused `model_wells` and `model_prod` tables that contain only the data needed for analog modeling.

## Step 1: Backup Current Data (Optional)
If you want to keep the existing data, create a backup first:
```sql
-- Create backup tables if needed
CREATE TABLE data.ed_wells_backup AS SELECT * FROM data.ed_wells;
CREATE TABLE data.ed_prod_backup AS SELECT * FROM data.ed_prod;
```

## Step 2: Clean Up Database
Run the cleanup script to remove large tables and create new structure:
```bash
psql -h minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com \
     -U minrights -d dde -f cleanup_database.sql
```

This will:
- Drop the massive `ed_wells` and `ed_prod` tables
- Create new `model_wells` and `model_prod` tables
- Add proper indexes

## Step 3: Upload Glue Script to S3
```bash
aws s3 cp model_data_etl.py s3://aws-glue-assets-277707132474-us-east-1/scripts/
```

## Step 4: Create/Update Glue Job
```bash
aws glue create-job --cli-input-json file://model-data-job-config.json
```

Or update existing job:
```bash
aws glue update-job --job-name model-data-etl \
                    --job-update file://model-data-job-config.json
```

## Step 5: Run the ETL Job
```bash
aws glue start-job-run --job-name model-data-etl
```

Monitor progress:
```bash
aws glue get-job-run --job-name model-data-etl \
                     --run-id <run-id-from-previous-command>
```

## Step 6: Verify Data
Once complete, verify the data:
```sql
-- Check record counts
SELECT COUNT(*) FROM data.model_wells;
SELECT COUNT(*) FROM data.model_prod;

-- Check basins imported
SELECT basin, COUNT(*) FROM data.model_wells GROUP BY basin;

-- Check date ranges
SELECT MIN(prod_date), MAX(prod_date) FROM data.model_prod;
```

## Step 7: Run Analog Model
Now you can run the analog model implementation:
```bash
cd /Users/billgivens/macmini/dde/dde-model
source venv/bin/activate

# Build early production table
python src/data/early_production.py

# Build candidate pools
python src/data/candidate_pool.py
```

## Configuration
The ETL job filters for:
- **Basins**: Denver Basin, Permian Basin, Eagle Ford (configurable)
- **Well Type**: Horizontal only
- **Time Period**: 2010 and newer
- **Quality**: Must have lateral length and location

To add more basins, update the `BASINS_FILTER` parameter in the job config.

## Benefits
- **Much smaller tables** - Only relevant data
- **Fast queries** - Proper indexing on focused data
- **Expandable** - Easy to add new basins
- **Clean** - No unnecessary historical data

## Estimated Data Sizes
- **model_wells**: ~50,000 wells across 3 major basins
- **model_prod**: ~10-20 million records (vs billions in original)
- **Query time**: Seconds instead of minutes