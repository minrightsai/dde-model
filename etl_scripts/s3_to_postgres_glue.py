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

logger.info(f"Starting ETL job: {args['JOB_NAME']}")
logger.info(f"S3 input path: {args['S3_INPUT_PATH']}")
logger.info(f"PostgreSQL target: {args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}")

try:
    # Step 1: Read parquet files from S3
    logger.info("Reading parquet files from S3...")
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
    
    logger.info("Successfully read data from S3")
    
    # Step 2: Convert to Spark DataFrame for transformations
    df = dynamic_frame.toDF()
    
    # Step 3: Filter out records with dates before 2000 to avoid Spark compatibility issues
    logger.info("Filtering out records before 2000...")
    df_filtered_dates = df.filter(
        (F.col("completion_date").isNull() | (F.col("completion_date") >= F.lit("2000-01-01"))) &
        (F.col("spud_date").isNull() | (F.col("spud_date") >= F.lit("2000-01-01"))) &
        (F.col("permit_date").isNull() | (F.col("permit_date") >= F.lit("2000-01-01")))
    )
    
    logger.info("Date filtering completed")
    
    # Step 4: Data cleaning and transformations
    logger.info("Applying data transformations...")
    
    # Clean and standardize data
    df_clean = df_filtered_dates.select(
        # Well identifiers
        F.col("well_api").alias("well_id"),
        F.col("universal_doc_no").alias("universal_doc_no"),
        
        # Location data  
        F.col("surface_lat").cast("double").alias("lat"),
        F.col("surface_lng").cast("double").alias("lon"),
        
        # Well metadata
        F.col("operator_name").alias("operator"),
        F.col("reservoir").alias("formation"),
        F.col("well_name").alias("well_name"),
        F.col("state_abbr").alias("state"),
        F.col("county").alias("county"),
        F.col("basin_name").alias("basin"),
        
        # Completion data
        F.col("completion_date").cast("timestamp").alias("first_prod_date"),
        F.col("lateral_length").cast("integer").alias("lateral_length"),
        F.col("proppant_lbs").cast("integer").alias("proppant_lbs"),
        F.col("water_gals").cast("double").alias("fluid_gals"),
        F.col("measured_depth").cast("integer").alias("measured_depth"),
        F.col("true_vertical_depth").cast("integer").alias("tvd"),
        
        # Additional metadata
        F.col("drill_type").alias("drill_type"),
        F.col("well_type").alias("well_type"),
        F.col("well_status").alias("well_status"),
        F.col("spud_date").cast("timestamp").alias("spud_date"),
        F.col("permit_date").cast("timestamp").alias("permit_date")
    )
    
    # Filter out rows with missing critical data
    df_filtered = df_clean.filter(
        F.col("well_id").isNotNull() &
        F.col("lat").isNotNull() & 
        F.col("lon").isNotNull() &
        F.col("first_prod_date").isNotNull()
    )
    
    # Calculate derived columns
    df_enhanced = df_filtered.withColumn(
        "proppant_per_ft", 
        F.when(F.col("lateral_length") > 0, 
               F.col("proppant_lbs") / F.col("lateral_length"))
        .otherwise(None)
    ).withColumn(
        "fluid_per_ft",
        F.when(F.col("lateral_length") > 0,
               F.col("fluid_gals") / F.col("lateral_length"))
        .otherwise(None)
    )
    
    logger.info("Data cleaning and enhancement completed")
    
    # Step 4: Convert back to DynamicFrame for writing
    dynamic_frame_clean = DynamicFrame.fromDF(df_enhanced, glueContext, "clean_data")
    
    # Step 5: Write to PostgreSQL
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
    
    logger.info("ETL job completed successfully")
    
except Exception as e:
    logger.error(f"ETL job failed: {str(e)}")
    raise e

finally:
    job.commit()