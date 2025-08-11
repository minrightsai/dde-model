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

# Get optional basin parameter (defaults to DJ Basin)
optional_args = getResolvedOptions(sys.argv, ['TARGET_BASINS'], optional=True)
target_basins = optional_args.get('TARGET_BASINS', 'Denver Basin').split(',')
logger.info(f"Target basins: {target_basins}")

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
logger.info(f"Processing basins: {', '.join(target_basins)}")

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
        F.lit(None).cast("double").alias("gas_mcf")
    ).filter(
        F.col("prod_date") >= F.lit("2000-01-01")  # Filter to modern wells only
    )
    
    gas_clean = gas_df.select(
        F.col("state_well_id").alias("well_id"), 
        F.col("production_date").alias("prod_date"),
        F.lit(None).cast("double").alias("oil_bbls"),
        F.col("production_quantity").alias("gas_mcf")
    ).filter(
        F.col("prod_date") >= F.lit("2000-01-01")  # Filter to modern wells only
    )
    
    logger.info("Data processing completed")
    
    # Step 5: Union oil and gas data
    logger.info("Merging oil and gas production data...")
    production_df = oil_clean.union(gas_clean)
    
    # Step 6: Filter to DJ Basin wells using geographic criteria (not well_type)
    logger.info("Reading model wells for DJ Basin filtering...")
    
    # Read model_wells table to get DJ Basin wells by geography, not well_type
    postgres_url = f"jdbc:postgresql://{args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}"
    
    dj_basin_wells_df = spark.read \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "data.model_wells") \
        .option("user", args['POSTGRES_USERNAME']) \
        .option("password", args['POSTGRES_PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .load() \
        .filter(F.col("basin").isin(target_basins)) \
        .select("well_id")
    
    logger.info("Filtering production data to DJ Basin wells with actual production...")
    # Filter production data to only include DJ Basin wells that have production data
    production_filtered = production_df.join(
        dj_basin_wells_df,
        on="well_id",
        how="leftsemi"  # Only keep production records for DJ Basin wells
    )
    
    # Step 7: Aggregate by well and date to combine oil and gas into single records
    logger.info("Aggregating production data by well and date...")
    production_agg = production_filtered.groupBy("well_id", "prod_date").agg(
        F.max("oil_bbls").alias("oil_bbls"),
        F.max("gas_mcf").alias("gas_mcf")
    )
    
    # Step 8: Calculate correct prod_month based on peak oil production
    logger.info("Calculating production month indices from peak production...")
    
    # Get first production date for each well
    window_spec = Window.partitionBy("well_id").orderBy("prod_date")
    
    production_with_first = production_agg.withColumn(
        "first_prod_date", F.first("prod_date").over(window_spec)
    )
    
    # Calculate initial months since first production (0-indexed)
    production_with_initial_month = production_with_first.withColumn(
        "initial_prod_month",
        F.floor(
            F.months_between(
                F.col("prod_date"),
                F.col("first_prod_date")
            )
        ).cast("bigint")
    )
    
    # Step 8a: Find peak oil production month (within first 7 months)
    logger.info("Finding peak oil production month for each well...")
    
    # Filter to first 7 months (0-6) and find peak
    first_7_months = production_with_initial_month.filter(
        F.col("initial_prod_month") <= 6
    ).filter(
        F.col("oil_bbls").isNotNull() & (F.col("oil_bbls") > 0)
    )
    
    # Use window function to find the actual peak month for each well
    window_spec_peak = Window.partitionBy("well_id").orderBy(F.col("oil_bbls").desc())
    
    # Add row number to identify the peak production month
    with_peak_rank = first_7_months.withColumn(
        "rank", F.row_number().over(window_spec_peak)
    )
    
    # Get only the peak month (rank = 1) for each well
    peak_months = with_peak_rank.filter(
        F.col("rank") == 1
    ).select(
        F.col("well_id"),
        F.col("initial_prod_month").alias("peak_month"),
        F.col("oil_bbls").alias("peak_oil_production")
    )
    
    # Log statistics about peak months
    peak_stats = peak_months.groupBy("peak_month").count().orderBy("peak_month").collect()
    logger.info(f"Peak month distribution: {peak_stats}")
    
    # Log additional statistics
    peak_month_summary = peak_months.agg(
        F.count("*").alias("total_wells"),
        F.avg("peak_month").alias("avg_peak_month"),
        F.avg("peak_oil_production").alias("avg_peak_oil")
    ).collect()[0]
    logger.info(f"Peak month summary - Total wells: {peak_month_summary['total_wells']}, "
                f"Avg peak month: {peak_month_summary['avg_peak_month']:.2f}, "
                f"Avg peak oil: {peak_month_summary['avg_peak_oil']:.0f} bbls")
    
    # All wells with identified peaks are valid (peak is within 0-6 by construction)
    valid_peak_wells = peak_months
    
    # Join back with production data
    production_with_peak = production_with_initial_month.join(
        valid_peak_wells,
        on="well_id",
        how="inner"  # Only keep wells with valid peak months
    )
    
    # Step 8b: Reset prod_month to start from peak
    logger.info("Resetting production month indices from peak...")
    
    # Calculate new prod_month (peak month becomes month 0)
    # Only keep data from peak month onwards
    production_peak_aligned = production_with_peak.filter(
        F.col("initial_prod_month") >= F.col("peak_month")
    ).withColumn(
        "prod_month",
        (F.col("initial_prod_month") - F.col("peak_month")).cast("bigint")
    )
    
    # Join with wells to get basin information
    wells_basin_df = spark.read \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "(SELECT well_id, basin, basin_name FROM data.model_wells) as wells") \
        .option("user", args['POSTGRES_USERNAME']) \
        .option("password", args['POSTGRES_PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .load()
    
    # Add basin information and finalize
    production_with_basin = production_peak_aligned.join(
        wells_basin_df,
        on="well_id",
        how="left"
    )
    
    # Remove temporary columns and add water/timestamp
    production_final = production_with_basin.select(
        "well_id",
        "prod_date",
        "prod_month",
        "oil_bbls",
        "gas_mcf",
        "basin",
        "basin_name"
    ).withColumn(
        "water_bbls", F.lit(None).cast("double")
    ).withColumn(
        "created_at", F.current_timestamp()
    )
    
    # Log final statistics
    well_count = production_final.select("well_id").distinct().count()
    record_count = production_final.count()
    logger.info(f"Final dataset: {well_count} wells, {record_count} production records")
    
    logger.info("Production data aggregation completed")
    
    # Step 9: Convert back to DynamicFrame
    production_dynamic_frame = DynamicFrame.fromDF(production_final, glueContext, "production_data")
    
    # Step 10: Clear existing data and write to PostgreSQL
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