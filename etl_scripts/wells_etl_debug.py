"""
Debug version of wells ETL to inspect actual schema from S3
Add this section after reading the parquet files to see what columns are available
"""

# Add this after line 60 where wells_df is created:

# Step 2: Convert to Spark DataFrame and INSPECT SCHEMA
wells_df = wells_dynamic_frame.toDF()
logger.info(f"Total wells read from S3: {wells_df.count():,}")

# DEBUG: Print actual schema from S3
logger.info("=" * 60)
logger.info("ACTUAL SCHEMA FROM S3 PARQUET FILES:")
logger.info("=" * 60)

# Print all columns
all_columns = wells_df.columns
logger.info(f"Total columns available: {len(all_columns)}")
logger.info("Column names:")
for col in sorted(all_columns):
    logger.info(f"  - {col}")

# Check for formation-related columns
formation_candidates = [col for col in all_columns if 'form' in col.lower() or 'zone' in col.lower() or 'target' in col.lower() or 'interval' in col.lower()]
if formation_candidates:
    logger.info(f"\nPOTENTIAL FORMATION COLUMNS FOUND: {formation_candidates}")
else:
    logger.info("\nNO FORMATION-RELATED COLUMNS FOUND")

# Show sample data for inspection
logger.info("\nSample data (first row):")
wells_df.show(1, truncate=False)

# Check data types
logger.info("\nSchema with data types:")
wells_df.printSchema()

logger.info("=" * 60)