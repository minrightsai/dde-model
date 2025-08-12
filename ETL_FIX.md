# ETL Pipeline Fix Documentation

## Overview
Complete redesign of the ETL pipeline to consistently process DJ Basin and Bakken data from S3 parquet files through to final tables. This fixes issues with inconsistent basin naming, data overwriting, and ensures only wells with actual production are loaded.

## Current Problems
1. **Inconsistent basin naming**: "Denver Basin" vs "BAKKEN", "dj_basin" vs "bakken"
2. **Different loading processes**: DJ loaded via Glue, Bakken loaded via data.wells (wrong)
3. **Data overwriting**: Production ETL overwrites all data instead of handling multiple basins
4. **Missing basin identification**: Production data lacks basin fields
5. **Unreliable dates**: Using metadata first_prod_date instead of actual production dates
6. **Loading non-producing wells**: Current process may load wells without production

## New Architecture

```
Stage 1: Wells ETL (Run First)
S3 Well Headers � Geographic + Date Filter (spud_date>=2014) � basin_wells_list

Stage 2: Production ETL (Run Second)  
S3 Production Data + basin_wells_list � Peak Alignment � model_prod

Stage 3: Wells Finalization
model_prod � Update model_wells.first_prod_date

Stage 4: Python Processing
model_wells + model_prod � early_production � candidate_pools � embeddings
```

## Two-Job Design

### Job 1: Wells ETL (`wells_etl_glue.py`)

**Purpose**: Load well headers with geographic and date filtering, create basin well list

**Inputs**:
- S3: `s3://.../well_combined/` (well headers parquet)

**Outputs**:
- `data.model_wells` - Well metadata with basin assignments (first_prod_date updated later)
- Well IDs list for Production ETL filtering

**Key Logic**:
```python
# 1. Read well headers from S3
well_headers = read_from_s3("s3://.../well_combined/")

# 2. Apply geographic and date filters upfront
# DJ Basin: Colorado wells in specific bounds, spud_date >= 2014
dj_filter = (
    (F.col("state_abbr") == "CO") & 
    (F.col("surface_lat").between(39.51, 42.19)) &
    (F.col("surface_lng").between(-105.13, -101.99)) &
    (F.col("spud_date") >= "2014-01-01")
)

# Bakken: ND/MT wells in specific bounds, spud_date >= 2014
bakken_filter = (
    (F.col("state_abbr").isin(["ND", "MT"])) &
    (F.col("surface_lat").between(46.66, 48.99)) &
    (F.col("surface_lng").between(-105.26, -102.00)) &
    (F.col("spud_date") >= "2014-01-01")
)

# 3. Filter to target basin wells only
basin_wells = well_headers.filter(dj_filter | bakken_filter)

# 4. Assign basin names and create model_wells
model_wells = basin_wells.withColumn(
    "basin",
    F.when(dj_filter, "dj")
     .when(bakken_filter, "bakken")
).withColumn(
    "basin_name", 
    F.when(F.col("basin") == "dj", "dj_basin")
     .when(F.col("basin") == "bakken", "bakken")
)

# 5. Write to model_wells (first_prod_date updated later)
```

### Job 2: Production ETL (`production_etl_glue.py`)

**Purpose**: Process production data for basin wells, write to model_prod

**Inputs**:
- S3: `s3://.../well_prod_liq_master/` (oil production parquet)
- S3: `s3://.../well_prod_gas_master/` (gas production parquet)
- PostgreSQL: `data.model_wells` (get well IDs from Job 1)

**Outputs**:
- `data.model_prod` - Production data with peak alignment and basin info

**Key Logic**:
```python
# 1. Get well IDs from model_wells (basin wells from Job 1)
basin_wells = spark.sql("SELECT well_id, basin, basin_name FROM data.model_wells")

# 2. Read and union oil/gas production
production_df = oil_df.union(gas_df)

# 3. Filter to basin wells only (huge data reduction!)
production_filtered = production_df.join(
    basin_wells.select("well_id"),
    production_df["state_well_id"] == basin_wells["well_id"],
    "inner"
).filter(
    F.col("production_date") >= "2014-01-01"
)

# 4. Identify wells with 9+ months production
wells_with_production = production_filtered.groupBy("state_well_id").agg(
    F.countDistinct("production_date").alias("months_count")
).filter(F.col("months_count") >= 9)

# 5. Filter to wells with sufficient production history
production_to_process = production_filtered.join(
    wells_with_production.select("state_well_id"),
    on="state_well_id",
    how="inner"
)

# 6. Join with basin info and apply peak alignment
production_with_basin = production_to_process.join(
    basin_wells,
    production_to_process["state_well_id"] == basin_wells["well_id"],
    "left"
)

# 7. Apply peak month detection and write to model_prod
```

**Peak Detection Logic**:
- Peak month window: 7 months (0-6) for all basins
- Consistent approach across all regions

## Basin Naming Convention

**Standardized across all tables**:
- `basin`: "dj" or "bakken" (short display names)
- `basin_name`: "dj_basin" or "bakken" (Python config keys)

This replaces the inconsistent "Denver Basin", "BAKKEN", etc.

## Geographic Basin Assignment

Instead of using API prefixes (which overlap), basins are assigned by geographic bounds:

### DJ Basin (Denver-Julesburg)
- State: Colorado
- Latitude: 39.51 to 42.19
- Longitude: -105.13 to -101.99
- Covers the northeastern Colorado oil fields

### Bakken
- States: North Dakota, Montana
- Latitude: 46.66 to 48.99
- Longitude: -105.26 to -102.00
- Covers the Williston Basin area

## Key Improvements

1. **Production-First Approach**: Only loads wells that actually have production
2. **Accurate Dates**: first_prod_date comes from actual production MIN(date), not metadata
3. **Consistent Basin Names**: Same format across all tables
4. **Geographic Filtering**: Precise basin assignment based on lat/lon
5. **No Data.Wells**: Direct S3 to model_wells, no intermediate table
6. **Two-Stage Process**: Production ETL identifies valid wells, Wells ETL loads metadata

## Execution Order

```bash
# 1. Clear tables (optional for fresh start)
psql -f truncate_all_tables.sql

# 2. Upload scripts to S3
aws s3 cp production_etl_glue.py s3://aws-glue-assets-.../scripts/
aws s3 cp wells_etl_glue.py s3://aws-glue-assets-.../scripts/

# 3. Run Wells ETL (30-45 minutes)
aws glue start-job-run --job-name wells-etl-job

# 4. Run Production ETL (60-90 minutes)
aws glue start-job-run --job-name production-etl-job

# 5. Update model_wells with actual production dates
psql -c "UPDATE data.model_wells SET first_prod_date = (SELECT MIN(prod_date) FROM data.model_prod WHERE model_prod.well_id = model_wells.well_id)"

# 6. Run Python processing
python -m src.data.early_production --basin dj_basin
python -m src.data.early_production --basin bakken
python -m src.data.candidate_pool --basin dj_basin
python -m src.data.candidate_pool --basin bakken
python -m src.features.embeddings
```

## AWS Glue Job Configurations

### Production ETL Job
```json
{
  "Name": "production-etl-job",
  "ScriptLocation": "s3://aws-glue-assets-.../scripts/production_etl_glue.py",
  "DefaultArguments": {
    "--S3_OIL_PATH": "s3://.../well_prod_liq_master/",
    "--S3_GAS_PATH": "s3://.../well_prod_gas_master/",
    "--POSTGRES_TABLE": "data.model_prod",
    "--TARGET_BASINS": "dj,bakken"
  },
  "MaxCapacity": 10
}
```

### Wells ETL Job
```json
{
  "Name": "wells-etl-job",
  "ScriptLocation": "s3://aws-glue-assets-.../scripts/wells_etl_glue.py",
  "DefaultArguments": {
    "--S3_WELLS_PATH": "s3://.../well_combined/",
    "--POSTGRES_TABLE": "data.model_wells"
  },
  "MaxCapacity": 10
}
```

## Critical Notes

1. **Well ID Mapping**: Verify that `state_well_id` (production) matches `well_api` (headers)
2. **Job Dependencies**: Production ETL must run after Wells ETL completes
3. **Overwrite Mode**: Both jobs use overwrite since we're starting fresh
4. **Peak Detection**: Uses 7 months (0-6) consistently for all basins
5. **No data.wells**: This table should NOT be used for anything

## Validation Queries

After running the pipeline, validate with:

```sql
-- Check basin consistency
SELECT basin, basin_name, COUNT(*) 
FROM data.model_wells 
GROUP BY basin, basin_name;

-- Verify wells and production match
SELECT 
    w.basin, w.basin_name, 
    COUNT(DISTINCT w.well_id) as wells, 
    COUNT(DISTINCT p.well_id) as prod_wells
FROM data.model_wells w
LEFT JOIN data.model_prod p ON w.well_id = p.well_id
GROUP BY w.basin, w.basin_name;

-- Check date consistency
SELECT w.well_id, w.first_prod_date, MIN(p.prod_date) as actual_min
FROM data.model_wells w
JOIN data.model_prod p ON w.well_id = p.well_id
GROUP BY w.well_id, w.first_prod_date
HAVING w.first_prod_date != MIN(p.prod_date)
LIMIT 10;  -- Should return 0 rows
```

## Next Steps

1. Implement basin-specific peak month windows
2. Add monitoring/alerting for job failures
3. Consider incremental updates instead of full overwrites
4. Add data quality checks between stages