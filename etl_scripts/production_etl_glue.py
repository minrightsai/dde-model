import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'S3_OIL_PATH',
    'S3_GAS_PATH', 
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

logger.info(f"Starting production ETL job: {args['JOB_NAME']}")
logger.info(f"Oil path: {args['S3_OIL_PATH']}")
logger.info(f"Gas path: {args['S3_GAS_PATH']}")
logger.info(f"PostgreSQL target: {args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}")

try:
    # Step 1: Read oil production data from S3
    logger.info("Reading oil production data from S3...")
    oil_dynamic_frame = glueContext.create_dynamic_frame.from_options(
        format_options={
            "int96RebaseMode": "LEGACY"
        },
        connection_type="s3",
        format="parquet",
        connection_options={
            "paths": [args['S3_OIL_PATH']],
            "recurse": True
        },
        transformation_ctx="oil_source"
    )
    
    # Step 2: Read gas production data from S3
    logger.info("Reading gas production data from S3...")
    gas_dynamic_frame = glueContext.create_dynamic_frame.from_options(
        format_options={
            "int96RebaseMode": "LEGACY"
        },
        connection_type="s3",
        format="parquet",
        connection_options={
            "paths": [args['S3_GAS_PATH']],
            "recurse": True
        },
        transformation_ctx="gas_source"
    )
    
    logger.info("Successfully read production data from S3")
    
    # Step 3: Convert to Spark DataFrames
    oil_df = oil_dynamic_frame.toDF()
    gas_df = gas_dynamic_frame.toDF()
    
    # Step 4: Add product type and clean data
    logger.info("Processing oil and gas data...")
    
    # Add product type to distinguish oil vs gas
    oil_clean = oil_df.select(
        F.col("state_well_id").alias("well_id"),
        F.col("production_date").alias("prod_date"),
        F.col("production_quantity").alias("oil_bbls"),
        F.lit(None).cast("double").alias("gas_mcf"),
        F.col("cum_production_quantity").cast("double").alias("cum_oil_bbls"),
        F.lit(None).cast("double").alias("cum_gas_mcf")
    ).filter(
        F.col("prod_date") >= F.lit("2000-01-01")  # Filter to modern wells only
    )
    
    gas_clean = gas_df.select(
        F.col("state_well_id").alias("well_id"), 
        F.col("production_date").alias("prod_date"),
        F.lit(None).cast("double").alias("oil_bbls"),
        F.col("production_quantity").alias("gas_mcf"),
        F.lit(None).cast("double").alias("cum_oil_bbls"),
        F.col("cum_production_quantity").cast("double").alias("cum_gas_mcf")
    ).filter(
        F.col("prod_date") >= F.lit("2000-01-01")  # Filter to modern wells only
    )
    
    logger.info("Data processing completed")
    
    # Step 5: Union oil and gas data
    logger.info("Merging oil and gas production data...")
    production_df = oil_clean.union(gas_clean)
    
    # Step 6: Aggregate by well and date to combine oil and gas into single records
    logger.info("Aggregating production data by well and date...")
    production_agg = production_df.groupBy("well_id", "prod_date").agg(
        F.max("oil_bbls").alias("oil_bbls"),
        F.max("gas_mcf").alias("gas_mcf"),
        F.max("cum_oil_bbls").alias("cum_oil_bbls"),
        F.max("cum_gas_mcf").alias("cum_gas_mcf")
    )
    
    # Step 7: Calculate correct prod_month based on first production date
    logger.info("Calculating production month indices...")
    
    # Get first production date for each well
    window_spec = Window.partitionBy("well_id").orderBy("prod_date")
    
    production_with_first = production_agg.withColumn(
        "first_prod_date", F.first("prod_date").over(window_spec)
    )
    
    # Calculate months since first production (0-indexed)
    production_with_month = production_with_first.withColumn(
        "prod_month",
        F.floor(
            F.months_between(
                F.col("prod_date"),
                F.col("first_prod_date")
            )
        ).cast("bigint")
    )
    
    # Remove temporary first_prod_date column and add water/timestamp
    production_final = production_with_month.select(
        "well_id",
        "prod_date",
        "prod_month",
        "oil_bbls",
        "gas_mcf",
        "cum_oil_bbls",
        "cum_gas_mcf"
    ).withColumn(
        "water_bbls", F.lit(None).cast("double")
    ).withColumn(
        "created_at", F.current_timestamp()
    )
    
    logger.info("Production data aggregation completed")
    
    # Step 8: Convert back to DynamicFrame
    production_dynamic_frame = DynamicFrame.fromDF(production_final, glueContext, "production_data")
    
    # Step 9: Clear existing data and write to PostgreSQL
    logger.info("Clearing existing data from target table...")
    
    # First, truncate the existing table using JDBC
    postgres_url = f"jdbc:postgresql://{args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}"
    
    # Create a connection to truncate the table
    production_final.write \
        .mode("overwrite") \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", args['POSTGRES_TABLE']) \
        .option("user", args['POSTGRES_USERNAME']) \
        .option("password", args['POSTGRES_PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .option("truncate", "true") \
        .save()
    
    logger.info(f"Table {args['POSTGRES_TABLE']} cleared and new data written")
    
    logger.info("Production ETL job completed successfully")
    
except Exception as e:
    logger.error(f"Production ETL job failed: {str(e)}")
    raise e

finally:
    job.commit()