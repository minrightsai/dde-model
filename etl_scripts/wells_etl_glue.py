"""
AWS Glue ETL: Load well headers from S3 parquet files to model_wells table
Supports multiple basins with consistent processing
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'S3_WELLS_PATH',
    'POSTGRES_HOST',
    'POSTGRES_PORT',
    'POSTGRES_DATABASE',
    'POSTGRES_USERNAME',
    'POSTGRES_PASSWORD',
    'TARGET_BASINS',  # Comma-separated list: "Denver Basin,Bakken"
    'BATCH_SIZE'
])

# Parse target basins
target_basins = [b.strip() for b in args['TARGET_BASINS'].split(',')]
logger.info(f"Target basins: {target_basins}")

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info(f"Starting wells ETL job: {args['JOB_NAME']}")
logger.info(f"S3 wells path: {args['S3_WELLS_PATH']}")
logger.info(f"Processing basins: {', '.join(target_basins)}")

try:
    # Step 1: Read well headers from S3
    logger.info("Reading well headers from S3...")
    wells_dynamic_frame = glueContext.create_dynamic_frame.from_options(
        format_options={
            "int96RebaseMode": "LEGACY"
        },
        connection_type="s3",
        format="parquet",
        connection_options={
            "paths": [args['S3_WELLS_PATH']],
            "recurse": True
        },
        transformation_ctx="wells_source"
    )
    
    # Step 2: Convert to Spark DataFrame
    wells_df = wells_dynamic_frame.toDF()
    logger.info(f"Total wells read: {wells_df.count():,}")
    
    # Step 3: Filter and clean data
    logger.info("Filtering and cleaning well data...")
    
    # Define basin filtering logic
    basin_filters = []
    for basin in target_basins:
        if basin == "Denver Basin":
            # DJ Basin: Colorado wells in specific lat/lon bounds
            basin_filters.append(
                (F.col("state_abbr") == "CO") & 
                (F.col("surface_lat").between(39.51, 42.19)) &
                (F.col("surface_lng").between(-105.13, -101.99))
            )
        elif basin == "Bakken":
            # Bakken: North Dakota and Montana wells in specific bounds
            basin_filters.append(
                (F.col("state_abbr").isin(["ND", "MT"])) &
                (F.col("surface_lat").between(46.66, 48.99)) &
                (F.col("surface_lng").between(-105.26, -102.00))
            )
    
    # Combine filters with OR
    if basin_filters:
        combined_filter = basin_filters[0]
        for f in basin_filters[1:]:
            combined_filter = combined_filter | f
        wells_filtered = wells_df.filter(combined_filter)
    else:
        wells_filtered = wells_df
    
    logger.info(f"Wells after basin filtering: {wells_filtered.count():,}")
    
    # Step 4: Transform and standardize
    wells_clean = wells_filtered.select(
        # Primary key
        F.col("well_api").alias("well_id"),
        
        # Well metadata
        F.col("well_name"),
        F.col("operator_name").alias("operator"),
        F.col("state_abbr").alias("state"),
        F.col("county"),
        
        # Basin assignment based on location
        F.when(
            (F.col("state_abbr") == "CO") & 
            (F.col("surface_lat").between(39.51, 42.19)) &
            (F.col("surface_lng").between(-105.13, -101.99)),
            F.lit("Denver Basin")
        ).when(
            (F.col("state_abbr").isin(["ND", "MT"])) &
            (F.col("surface_lat").between(46.66, 48.99)) &
            (F.col("surface_lng").between(-105.26, -102.00)),
            F.lit("Bakken")
        ).otherwise(F.lit(None)).alias("basin"),
        
        # Basin name for Python config
        F.when(
            (F.col("state_abbr") == "CO") & 
            (F.col("surface_lat").between(39.51, 42.19)) &
            (F.col("surface_lng").between(-105.13, -101.99)),
            F.lit("dj_basin")
        ).when(
            (F.col("state_abbr").isin(["ND", "MT"])) &
            (F.col("surface_lat").between(46.66, 48.99)) &
            (F.col("surface_lng").between(-105.26, -102.00)),
            F.lit("bakken")
        ).otherwise(F.lit(None)).alias("basin_name"),
        
        F.col("sub_basin"),
        
        # Location
        F.col("surface_lat").cast("double").alias("lat"),
        F.col("surface_lng").cast("double").alias("lon"),
        
        # Dates
        F.col("spud_date").cast("date"),
        F.col("completion_date").cast("date"),
        F.col("first_prod_date").cast("date"),
        
        # Well type and trajectory
        F.col("primary_fluid").alias("well_type"),
        F.col("trajectory").alias("drill_type"),
        F.col("formation"),
        
        # Completion data
        F.col("lateral_length").cast("double"),
        F.col("proppant_lbs").cast("double"),
        F.col("water_gals").cast("double").alias("fluid_gals"),
        
        # Timestamps
        F.current_timestamp().alias("created_at")
    ).filter(
        # Quality filters
        (F.col("drill_type") == "HORIZONTAL") &  # Horizontal wells only
        (F.col("first_prod_date") >= F.lit("2010-01-01")) &  # Modern wells
        (F.col("lat").isNotNull()) &
        (F.col("lon").isNotNull()) &
        (F.col("lateral_length") > 0) &
        (F.col("basin").isNotNull())  # Must have basin assignment
    )
    
    logger.info(f"Wells after quality filtering: {wells_clean.count():,}")
    
    # Step 5: Write to PostgreSQL
    logger.info("Writing to model_wells table...")
    
    postgres_url = f"jdbc:postgresql://{args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}"
    
    # Write with overwrite mode (since we're starting fresh)
    wells_clean.write \
        .mode("overwrite") \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "data.model_wells") \
        .option("user", args['POSTGRES_USERNAME']) \
        .option("password", args['POSTGRES_PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .option("batchsize", args.get('BATCH_SIZE', '10000')) \
        .save()
    
    # Log statistics
    basin_stats = wells_clean.groupBy("basin", "basin_name").count().collect()
    logger.info("Wells loaded by basin:")
    for row in basin_stats:
        logger.info(f"  {row['basin']} ({row['basin_name']}): {row['count']:,} wells")
    
    logger.info("Wells ETL job completed successfully")
    
except Exception as e:
    logger.error(f"Wells ETL job failed: {str(e)}")
    raise e

finally:
    job.commit()