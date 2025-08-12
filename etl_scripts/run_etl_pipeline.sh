#!/bin/bash

# ETL Pipeline Execution Script
# Runs the complete wells-first ETL pipeline

set -e  # Exit on any error

echo "Starting ETL Pipeline - Wells-First Approach"
echo "============================================="

# Configuration
WELLS_JOB_NAME="wells-etl-job"
PRODUCTION_JOB_NAME="production-etl-job"
POSTGRES_HOST="minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com"
POSTGRES_DB="dde"
POSTGRES_USER="minrights"

# Step 1: Optional - Clear existing tables
echo ""
echo "Step 1: Clear existing tables (optional)"
read -p "Do you want to clear existing model_wells and model_prod tables? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing tables..."
    psql -h $POSTGRES_HOST -d $POSTGRES_DB -U $POSTGRES_USER -c "
        TRUNCATE TABLE data.model_wells, data.model_prod CASCADE;
        SELECT 'Tables cleared' as status;
    "
    echo "✓ Tables cleared"
else
    echo "Skipping table clear"
fi

# Step 2: Upload scripts to S3
echo ""
echo "Step 2: Uploading scripts to S3..."
aws s3 cp wells_etl_glue_new.py s3://aws-glue-assets-277707132474-us-east-1/scripts/
aws s3 cp production_etl_glue_new.py s3://aws-glue-assets-277707132474-us-east-1/scripts/
echo "✓ Scripts uploaded to S3"

# Step 3: Run Wells ETL (first job)
echo ""
echo "Step 3: Running Wells ETL Job..."
echo "Expected duration: 30-45 minutes"
WELLS_RUN_ID=$(aws glue start-job-run --job-name $WELLS_JOB_NAME --query 'JobRunId' --output text)
echo "Wells ETL started with run ID: $WELLS_RUN_ID"

# Wait for Wells ETL to complete
echo "Waiting for Wells ETL to complete..."
while true; do
    WELLS_STATUS=$(aws glue get-job-run --job-name $WELLS_JOB_NAME --run-id $WELLS_RUN_ID --query 'JobRun.JobRunState' --output text)
    echo "Wells ETL Status: $WELLS_STATUS"
    
    if [ "$WELLS_STATUS" = "SUCCEEDED" ]; then
        echo "✓ Wells ETL completed successfully"
        break
    elif [ "$WELLS_STATUS" = "FAILED" ] || [ "$WELLS_STATUS" = "STOPPED" ] || [ "$WELLS_STATUS" = "TIMEOUT" ]; then
        echo "❌ Wells ETL failed with status: $WELLS_STATUS"
        echo "Check AWS Glue console for error details"
        exit 1
    fi
    
    sleep 60  # Check every minute
done

# Step 4: Run Production ETL (second job)
echo ""
echo "Step 4: Running Production ETL Job..."
echo "Expected duration: 60-90 minutes"
PROD_RUN_ID=$(aws glue start-job-run --job-name $PRODUCTION_JOB_NAME --query 'JobRunId' --output text)
echo "Production ETL started with run ID: $PROD_RUN_ID"

# Wait for Production ETL to complete
echo "Waiting for Production ETL to complete..."
while true; do
    PROD_STATUS=$(aws glue get-job-run --job-name $PRODUCTION_JOB_NAME --run-id $PROD_RUN_ID --query 'JobRun.JobRunState' --output text)
    echo "Production ETL Status: $PROD_STATUS"
    
    if [ "$PROD_STATUS" = "SUCCEEDED" ]; then
        echo "✓ Production ETL completed successfully"
        break
    elif [ "$PROD_STATUS" = "FAILED" ] || [ "$PROD_STATUS" = "STOPPED" ] || [ "$PROD_STATUS" = "TIMEOUT" ]; then
        echo "❌ Production ETL failed with status: $PROD_STATUS"
        echo "Check AWS Glue console for error details"
        exit 1
    fi
    
    sleep 120  # Check every 2 minutes (longer job)
done

# Step 5: Update first_prod_date in model_wells
echo ""
echo "Step 5: Updating first_prod_date in model_wells..."
psql -h $POSTGRES_HOST -d $POSTGRES_DB -U $POSTGRES_USER -f update_first_prod_dates.sql
echo "✓ First production dates updated"

# Step 6: Validation
echo ""
echo "Step 6: Pipeline Validation"
echo "==========================="
psql -h $POSTGRES_HOST -d $POSTGRES_DB -U $POSTGRES_USER -c "
-- Final pipeline statistics
SELECT 'Pipeline Summary:' as status;

-- Wells by basin
SELECT 
    'Wells by basin' as metric,
    basin, 
    basin_name, 
    COUNT(*) as count
FROM data.model_wells 
GROUP BY basin, basin_name
ORDER BY basin;

-- Production records by basin  
SELECT 
    'Production records by basin' as metric,
    basin,
    basin_name,
    COUNT(DISTINCT well_id) as wells,
    COUNT(*) as records
FROM data.model_prod
GROUP BY basin, basin_name
ORDER BY basin;

-- Data quality check
SELECT 
    'Data Quality Check' as metric,
    COUNT(DISTINCT w.well_id) as wells_in_model_wells,
    COUNT(DISTINCT p.well_id) as wells_in_model_prod,
    COUNT(DISTINCT w.well_id) - COUNT(DISTINCT p.well_id) as wells_without_production
FROM data.model_wells w
FULL OUTER JOIN data.model_prod p ON w.well_id = p.well_id;
"

echo ""
echo "✓ ETL Pipeline completed successfully!"
echo ""
echo "Next steps:"
echo "1. python -m src.data.early_production --basin dj_basin"
echo "2. python -m src.data.early_production --basin bakken"
echo "3. python -m src.data.candidate_pool --basin dj_basin"
echo "4. python -m src.data.candidate_pool --basin bakken"
echo "5. python -m src.features.embeddings"