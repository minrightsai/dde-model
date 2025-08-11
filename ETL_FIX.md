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
Stage 1: Production ETL (Run First)
S3 Production Data ’ Filter (>2010) ’ Peak Alignment ’ model_prod
                                    “
                            production_wells (temp table with well IDs)

Stage 2: Wells ETL (Run Second)  
S3 Well Headers + production_wells ’ Geographic Filter ’ model_wells

Stage 3: Python Processing
model_wells + model_prod ’ early_production ’ candidate_pools ’ embeddings
```

## Two-Job Design

### Job 1: Production ETL (`production_etl_glue.py`)

**Purpose**: Process production data, identify wells with sufficient history, write to model_prod

**Inputs**:
- S3: `s3://.../well_prod_liq_master/` (oil production parquet)
- S3: `s3://.../well_prod_gas_master/` (gas production parquet)
- PostgreSQL: `data.model_wells` (for basin filtering)

**Outputs**:
- `data.model_prod` - Production data with peak alignment and basin fields
- `data.production_wells` - Temp table with well IDs that have production

**Key Logic**:
```python
# 1. Read and union oil/gas production
production_df = oil_df.union(gas_df)

# 2. Filter to modern wells only
production_filtered = production_df.filter(
    F.col("production_date") >= "2010-01-01"
)

# 3. Identify wells with 9+ months production
wells_with_production = production_filtered.groupBy("state_well_id").agg(
    F.min("production_date").alias("first_prod_date"),  # ACTUAL first prod date
    F.countDistinct("production_date").alias("months_count")
).filter(F.col("months_count") >= 9)

# 4. Write well list to temp table
wells_with_production.write.mode("overwrite").saveAsTable("data.production_wells")

# 5. Filter production to these wells
production_to_process = production_filtered.join(
    wells_with_production.select("state_well_id"),
    on="state_well_id",
    how="inner"
)

# 6. Apply peak month detection (months 0-6)
# 7. Reset production months from peak (peak = month 0)
# 8. Join with model_wells to get basin information
# 9. Write to model_prod with basin and basin_name fields
```

**Basin-Specific Logic**:
- Peak month window: 7 months for DJ, 6 months for Bakken
- Currently hardcoded to 7 (needs fixing for basin-specific)

### Job 2: Wells ETL (`wells_etl_glue.py`)

**Purpose**: Load well headers for wells that have production, assign basins based on geography

**Inputs**:
- S3: `s3://.../well_combined/` (well headers parquet)
- PostgreSQL: `data.production_wells` (from Job 1)

**Outputs**:
- `data.model_wells` - Well metadata with basin assignments

**Key Logic**:
```python
# 1. Read well IDs from production_wells temp table
prod_wells = spark.table("data.production_wells")
# Has: state_well_id, first_prod_date, months_count

# 2. Read well headers from S3
well_headers = read_from_s3("s3://.../well_combined/")

# 3. Inner join - only wells with production
wells_with_prod = well_headers.join(
    prod_wells,
    well_headers["well_api"] == prod_wells["state_well_id"],
    "inner"
)

# 4. Apply geographic filters for basin assignment
# DJ Basin: Colorado wells in lat 39.51-42.19, lon -105.13 to -101.99
dj_filter = (
    (F.col("state_abbr") == "CO") & 
    (F.col("surface_lat").between(39.51, 42.19)) &
    (F.col("surface_lng").between(-105.13, -101.99))
)

# Bakken: ND/MT wells in lat 46.66-48.99, lon -105.26 to -102.00
bakken_filter = (
    (F.col("state_abbr").isin(["ND", "MT"])) &
    (F.col("surface_lat").between(46.66, 48.99)) &
    (F.col("surface_lng").between(-105.26, -102.00))
)

# 5. Assign basin names
wells_with_basin = wells_with_prod.withColumn(
    "basin",
    F.when(dj_filter, "dj")
     .when(bakken_filter, "bakken")
     .otherwise(None)
).filter(F.col("basin").isNotNull())

# 6. Use first_prod_date from production data (not metadata)
wells_final = wells_with_basin.select(
    F.col("well_api").alias("well_id"),
    # ... other well header fields ...
    F.col("first_prod_date"),  # From production_wells, not well headers!
    F.col("basin"),
    F.when(F.col("basin") == "dj", "dj_basin")
     .when(F.col("basin") == "bakken", "bakken")
     .alias("basin_name")
)

# 7. Write to model_wells
wells_final.write.mode("overwrite").saveAsTable("data.model_wells")
```

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

# 3. Run Production ETL (60-90 minutes)
aws glue start-job-run --job-name production-etl-job

# 4. Run Wells ETL (30-45 minutes)  
aws glue start-job-run --job-name wells-etl-job

# 5. Run Python processing
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
2. **Temp Table**: `data.production_wells` must persist between Job 1 and Job 2
3. **Overwrite Mode**: Both jobs use overwrite since we're starting fresh
4. **Peak Detection**: Currently uses 7 months for all basins - needs basin-specific logic
5. **No data.wells**: This table should NOT be used for anything

## Validation Queries

After running the pipeline, validate with:

```sql
-- Check basin consistency
SELECT basin, basin_name, COUNT(*) 
FROM data.model_wells 
GROUP BY basin, basin_name;

-- Verify production has basin info
SELECT basin, basin_name, COUNT(DISTINCT well_id) as wells, COUNT(*) as records
FROM data.model_prod
GROUP BY basin, basin_name;

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