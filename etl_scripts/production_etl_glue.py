"""
AWS Glue ETL: Production Processing (Wells-First Approach)
Process production data for basin wells identified by Wells ETL
This runs SECOND after Wells ETL creates the basin well filter list
"""

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
    'BATCH_SIZE'
])

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info(f"Starting Production ETL job: {args['JOB_NAME']}")
logger.info(f"Oil path: {args['S3_OIL_PATH']}")
logger.info(f"Gas path: {args['S3_GAS_PATH']}")

try:
    # Step 1: Get basin well IDs from model_wells (created by Wells ETL)
    logger.info("Reading basin well IDs from model_wells...")
    postgres_url = f"jdbc:postgresql://{args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}"
    
    basin_wells_df = spark.read \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "data.model_wells") \
        .option("user", args['POSTGRES_USERNAME']) \
        .option("password", args['POSTGRES_PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .load() \
        .select("well_id", "basin", "basin_name")  # Get basin info for later join
    
    basin_well_count = basin_wells_df.count()
    logger.info(f"Basin wells to process: {basin_well_count:,}")
    
    # Log basin distribution
    basin_distribution = basin_wells_df.groupBy("basin", "basin_name").count().collect()
    logger.info("Basin well distribution:")
    for row in basin_distribution:
        logger.info(f"  {row['basin']} ({row['basin_name']}): {row['count']:,} wells")
    
    # Step 2: Read oil production data from S3
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
    
    # Step 3: Read gas production data from S3
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
    
    # Step 4: Convert to Spark DataFrames and clean
    oil_df = oil_dynamic_frame.toDF()
    gas_df = gas_dynamic_frame.toDF()
    
    logger.info("Processing and cleaning production data...")
    
    # Clean and standardize oil data
    oil_clean = oil_df.select(
        F.col("state_well_id").alias("well_id"),
        F.col("production_date").alias("prod_date"),
        F.col("production_quantity").alias("oil_bbls"),
        F.lit(None).cast("double").alias("gas_mcf")
    ).filter(
        F.col("prod_date") >= F.lit("2014-01-01")  # Filter early for performance
    )
    
    # Clean and standardize gas data
    gas_clean = gas_df.select(
        F.col("state_well_id").alias("well_id"), 
        F.col("production_date").alias("prod_date"),
        F.lit(None).cast("double").alias("oil_bbls"),
        F.col("production_quantity").alias("gas_mcf")
    ).filter(
        F.col("prod_date") >= F.lit("2014-01-01")  # Filter early for performance
    )
    
    # Step 5: Union oil and gas data
    logger.info("Merging oil and gas production data...")
    production_df = oil_clean.union(gas_clean)
    
    # Step 6: MASSIVE data reduction - filter to basin wells only
    logger.info("Filtering production to basin wells only (major data reduction)...")
    production_filtered = production_df.join(
        basin_wells_df.select("well_id"),
        on="well_id",
        how="inner"  # Only keep production for basin wells
    )
    
    logger.info("Production data filtered to basin wells")
    
    # Step 7: Aggregate by well and date to combine oil and gas into single records
    logger.info("Aggregating production data by well and date...")
    production_agg = production_filtered.groupBy("well_id", "prod_date").agg(
        F.max("oil_bbls").alias("oil_bbls"),
        F.max("gas_mcf").alias("gas_mcf")
    )
    
    # Step 8: Identify wells with sufficient production history (9+ months)
    logger.info("Identifying wells with 9+ months of production...")
    
    wells_with_production = production_agg.groupBy("well_id").agg(
        F.countDistinct("prod_date").alias("months_count")
    ).filter(F.col("months_count") >= 9)
    
    valid_well_count = wells_with_production.count()
    logger.info(f"Wells with 9+ months production: {valid_well_count:,}")
    
    # Filter production to wells with sufficient history
    production_sufficient = production_agg.join(
        wells_with_production.select("well_id"),
        on="well_id",
        how="inner"
    )
    
    # Step 9: Calculate peak month alignment
    logger.info("Calculating production month indices and peak alignment...")
    
    # Get first production date for each well
    window_spec = Window.partitionBy("well_id").orderBy("prod_date")
    
    production_with_first = production_sufficient.withColumn(
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
    
    # Step 10: Find peak oil production month (within first 7 months: 0-6)
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
    
    # Log peak month statistics
    peak_stats = peak_months.groupBy("peak_month").count().orderBy("peak_month").collect()
    logger.info(f"Peak month distribution: {peak_stats}")
    
    # Join back with production data
    production_with_peak = production_with_initial_month.join(
        peak_months,
        on="well_id",
        how="inner"  # Only keep wells with valid peak months
    )
    
    # Step 11: Reset prod_month to start from peak (peak month becomes month 0)
    logger.info("Resetting production month indices from peak...")
    
    # Calculate new prod_month (peak month becomes month 0)
    # Only keep data from peak month onwards
    production_peak_aligned = production_with_peak.filter(
        F.col("initial_prod_month") >= F.col("peak_month")
    ).withColumn(
        "prod_month",
        (F.col("initial_prod_month") - F.col("peak_month")).cast("bigint")
    )
    
    # Step 12: Add basin information and finalize
    logger.info("Adding basin information and finalizing dataset...")
    
    # Join with basin information
    production_with_basin = production_peak_aligned.join(
        basin_wells_df.select("well_id", "basin", "basin_name"),
        on="well_id",
        how="left"
    )
    
    # Final data selection
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
    logger.info(f"Final dataset: {well_count:,} wells, {record_count:,} production records")
    
    # Basin statistics
    final_basin_stats = production_final.groupBy("basin", "basin_name").agg(
        F.countDistinct("well_id").alias("wells"),
        F.count("*").alias("records")
    ).collect()
    
    logger.info("Final production data by basin:")
    for row in final_basin_stats:
        logger.info(f"  {row['basin']} ({row['basin_name']}): {row['wells']:,} wells, {row['records']:,} records")
    
    # Step 13: Write to PostgreSQL
    logger.info("Writing to model_prod table...")
    
    production_final.write \
        .mode("overwrite") \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "data.model_prod") \
        .option("user", args['POSTGRES_USERNAME']) \
        .option("password", args['POSTGRES_PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .option("batchsize", args.get('BATCH_SIZE', '10000')) \
        .save()
    
    # Step 14: Update first_prod_date in model_wells from actual production data
    logger.info("Updating first_prod_date in model_wells from production data...")
    
    # Calculate first production date for each well
    first_prod_dates = production_final.groupBy("well_id").agg(
        F.min("prod_date").alias("first_prod_date")
    )
    
    wells_to_update = first_prod_dates.count()
    logger.info(f"Calculated first production dates for {wells_to_update:,} wells")
    
    # Write first_prod_dates to a temporary table in PostgreSQL
    first_prod_dates.write \
        .mode("overwrite") \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "temp_first_prod_dates") \
        .option("user", args['POSTGRES_USERNAME']) \
        .option("password", args['POSTGRES_PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .save()
    
    # Execute UPDATE using py4j direct JDBC connection
    from py4j.java_gateway import java_import
    
    # Import Java SQL classes
    java_import(sc._gateway.jvm, "java.sql.Connection")
    java_import(sc._gateway.jvm, "java.sql.DriverManager")
    java_import(sc._gateway.jvm, "java.sql.SQLException")
    
    # Create direct JDBC connection
    jdbc_url = f"jdbc:postgresql://{args['POSTGRES_HOST']}:{args['POSTGRES_PORT']}/{args['POSTGRES_DATABASE']}"
    conn = sc._gateway.jvm.DriverManager.getConnection(
        jdbc_url,
        args['POSTGRES_USERNAME'], 
        args['POSTGRES_PASSWORD']
    )
    
    # Execute UPDATE statement
    stmt = conn.createStatement()
    update_query = """
    UPDATE data.model_wells 
    SET first_prod_date = temp.first_prod_date
    FROM temp_first_prod_dates temp
    WHERE data.model_wells.well_id = temp.well_id
    """
    
    rows_updated = stmt.executeUpdate(update_query)
    logger.info(f"Updated first_prod_date for {rows_updated:,} wells")
    
    # Clean up temp table
    cleanup_query = "DROP TABLE IF EXISTS temp_first_prod_dates"
    stmt.executeUpdate(cleanup_query)
    
    # Close connection
    stmt.close()
    conn.close()
    
    logger.info(f"✅ first_prod_date updated successfully for {wells_to_update:,} wells in model_wells")
    logger.info("Production ETL job completed successfully")
    
except Exception as e:
    logger.error(f"Production ETL job failed: {str(e)}")
    raise e

finally:
    job.commit()