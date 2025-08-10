import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'S3_INPUT_PATH',
    'POSTGRES_HOST', 
    'POSTGRES_PORT',
    'POSTGRES_DATABASE',
    'POSTGRES_USERNAME',
    'POSTGRES_PASSWORD',
    'POSTGRES_TABLE',
    'BATCH_SIZE'
])

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info(f"Starting rigs ETL job: {args['JOB_NAME']}")
logger.info(f"S3 input path: {args['S3_INPUT_PATH']}")
logger.info(f"PostgreSQL target: {args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}")

try:
    # Step 1: Read parquet files from S3
    logger.info("Reading rigs data from S3...")
    dynamic_frame = glueContext.create_dynamic_frame.from_options(
        format_options={
            "int96RebaseMode": "LEGACY"
        },
        connection_type="s3",
        format="parquet",
        connection_options={
            "paths": [args['S3_INPUT_PATH']],
            "recurse": True
        },
        transformation_ctx="s3_source"
    )
    
    logger.info("Successfully read rigs data from S3")
    
    # Step 2: Convert to Spark DataFrame for transformations
    df = dynamic_frame.toDF()
    
    # Step 3: Filter out records with dates before 2000 to avoid Spark compatibility issues
    logger.info("Filtering out records before 2000...")
    df_filtered_dates = df.filter(
        (F.col("spud_date").isNull() | (F.col("spud_date") >= F.lit("2000-01-01"))) &
        (F.col("permit_date").isNull() | (F.col("permit_date") >= F.lit("2000-01-01")))
    )
    
    logger.info("Date filtering completed")
    
    # Step 4: Data cleaning and transformations
    logger.info("Applying data transformations...")
    
    # Clean and standardize rigs data
    df_clean = df_filtered_dates.select(
        # Rig identifiers
        F.col("rig_id").alias("rig_id"),
        F.col("rig_location_id").alias("rig_location_id"),
        F.col("name").alias("rig_name"),
        
        # Location data  
        F.col("latitude").cast("double").alias("lat"),
        F.col("longitude").cast("double").alias("lon"),
        
        # Rig metadata
        F.col("operator_name").alias("operator"),
        F.col("driller").alias("driller"),
        F.col("well_api").alias("well_id"),
        F.col("state_abbr").alias("state"),
        F.col("county_name").alias("county"),
        F.col("basin_name").alias("basin"),
        
        # Dates
        F.col("last_reported").cast("timestamp").alias("last_reported"),
        F.col("spud_date").cast("timestamp").alias("spud_date"),
        F.col("permit_date").cast("timestamp").alias("permit_date"),
        
        # Additional metadata
        F.col("drill_type").alias("drill_type"),
        F.col("reservoir").alias("reservoir"),
        F.col("source").alias("source"),
        F.col("source_type").alias("source_type"),
        F.col("pad_reported").alias("pad_reported"),
        F.col("coords_source").alias("coords_source")
    )
    
    # Filter out rows with missing critical data
    df_filtered = df_clean.filter(
        F.col("rig_id").isNotNull() &
        F.col("lat").isNotNull() & 
        F.col("lon").isNotNull()
    )
    
    # Add created timestamp
    df_enhanced = df_filtered.withColumn(
        "created_at", F.current_timestamp()
    ).withColumn(
        "status", F.lit("active")  # Assume all current rigs are active
    )
    
    logger.info("Data cleaning and enhancement completed")
    
    # Step 5: Convert back to DynamicFrame for writing
    dynamic_frame_clean = DynamicFrame.fromDF(df_enhanced, glueContext, "clean_data")
    
    # Step 6: Write to PostgreSQL
    logger.info("Writing to PostgreSQL...")
    
    # PostgreSQL connection options
    postgres_options = {
        "url": f"jdbc:postgresql://{args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}",
        "user": args['POSTGRES_USERNAME'],
        "password": args['POSTGRES_PASSWORD'],
        "dbtable": args['POSTGRES_TABLE'],
        "driver": "org.postgresql.Driver"
    }
    
    # Write data to PostgreSQL
    glueContext.write_dynamic_frame.from_options(
        frame=dynamic_frame_clean,
        connection_type="jdbc",
        connection_options=postgres_options,
        transformation_ctx="postgres_sink"
    )
    
    logger.info("Rigs ETL job completed successfully")
    
except Exception as e:
    logger.error(f"Rigs ETL job failed: {str(e)}")
    raise e

finally:
    job.commit()