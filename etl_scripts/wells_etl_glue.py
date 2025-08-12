"""
AWS Glue ETL: Wells-First Approach
Load well headers from S3, apply geographic and date filters, create model_wells table
This runs FIRST to create the basin well list for Production ETL filtering
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
    'BATCH_SIZE'
])

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info(f"Starting Wells ETL job: {args['JOB_NAME']}")
logger.info(f"S3 wells path: {args['S3_WELLS_PATH']}")

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
    logger.info(f"Total wells read from S3: {wells_df.count():,}")
    
    # Step 3: Apply geographic and date filters upfront for massive data reduction
    logger.info("Applying geographic and date filters...")
    
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
    
    # Filter to target basin wells only
    basin_wells = wells_df.filter(dj_filter | bakken_filter)
    logger.info(f"Wells after geographic+date filtering: {basin_wells.count():,}")
    
    # Step 4: Assign basin names and transform data
    logger.info("Assigning basin names and transforming data...")
    
    wells_with_basin = basin_wells.withColumn(
        "basin",
        F.when(dj_filter, "dj")
         .when(bakken_filter, "bakken")
         .otherwise(None)
    ).withColumn(
        "basin_name", 
        F.when(F.col("basin") == "dj", "dj_basin")
         .when(F.col("basin") == "bakken", "bakken")
         .otherwise(None)
    ).filter(F.col("basin").isNotNull())
    
    # Step 5: Transform and standardize fields
    wells_clean = wells_with_basin.select(
        # Primary key
        F.col("well_api").alias("well_id"),
        
        # Well metadata
        F.col("well_name"),
        F.col("operator_name").alias("operator"),
        F.col("state_abbr").alias("state"),
        F.col("county"),
        
        # Basin assignment (from geographic filtering)
        F.col("basin"),
        F.col("basin_name"),
        F.lit(None).cast("string").alias("sub_basin"),  # Not available in source data
        
        # Location
        F.col("surface_lat").cast("double").alias("lat"),
        F.col("surface_lng").cast("double").alias("lon"),
        
        # Dates (first_prod_date will be updated later from actual production)
        F.col("spud_date").cast("date"),
        F.col("completion_date").cast("date"),
        F.lit(None).cast("date").alias("first_prod_date"),  # Will be updated later
        
        # Well type and trajectory  
        F.col("well_type"),
        F.col("drill_type"),
        F.lit(None).cast("string").alias("formation"),  # Not available in source data
        
        # Completion data
        F.col("lateral_length").cast("integer"),
        F.col("proppant_lbs").cast("integer"),
        F.col("water_gals").cast("double").alias("fluid_gals"),
        
        # Timestamps
        F.current_timestamp().alias("created_at")
    )
    
    logger.info(f"Wells after transformations: {wells_clean.count():,}")
    
    # Step 6: Write to PostgreSQL
    logger.info("Writing to model_wells table...")
    
    postgres_url = f"jdbc:postgresql://{args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}"
    
    # Write with overwrite mode (fresh start)
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
    
    total_wells = wells_clean.count()
    logger.info(f"Total wells in model_wells: {total_wells:,}")
    logger.info("Wells ETL job completed successfully")
    
except Exception as e:
    logger.error(f"Wells ETL job failed: {str(e)}")
    raise e

finally:
    job.commit()