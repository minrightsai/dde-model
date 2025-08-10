#!/usr/bin/env python3
"""
Check if data has been imported to model tables
"""

import psycopg2
import boto3

conn_params = {
    'host': 'minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'dde',
    'user': 'minrights',
    'password': '2Knowthyself33!',
    'connect_timeout': 10
}

def check_tables():
    """Check model tables for data"""
    
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    print("Checking Model Tables")
    print("="*60)
    
    # Check model_wells
    cur.execute("SELECT COUNT(*) FROM data.model_wells")
    wells_count = cur.fetchone()[0]
    print(f"\ndata.model_wells: {wells_count:,} records")
    
    if wells_count > 0:
        # Get sample
        cur.execute("""
            SELECT basin, COUNT(*) 
            FROM data.model_wells 
            GROUP BY basin
            ORDER BY COUNT(*) DESC
        """)
        basins = cur.fetchall()
        print("\nWells by basin:")
        for basin, count in basins:
            print(f"  {basin}: {count:,}")
    
    # Check model_prod
    cur.execute("SELECT COUNT(*) FROM data.model_prod")
    prod_count = cur.fetchone()[0]
    print(f"\ndata.model_prod: {prod_count:,} records")
    
    if prod_count > 0:
        cur.execute("""
            SELECT MIN(prod_date), MAX(prod_date)
            FROM data.model_prod
        """)
        min_date, max_date = cur.fetchone()
        print(f"  Date range: {min_date} to {max_date}")
    
    cur.close()
    conn.close()
    
    return wells_count, prod_count

def check_job_status():
    """Check Glue job status"""
    
    glue = boto3.client('glue', region_name='us-east-1')
    
    response = glue.get_job_run(
        JobName='model-data-etl',
        RunId='jr_c62a7c26ecd2d036ef7a3effd69f65bf82ee199e9c2d0ff3cad8dcf4056627d1'
    )
    
    status = response['JobRun']['JobRunState']
    exec_time = response['JobRun'].get('ExecutionTime', 0)
    
    print("\n" + "="*60)
    print(f"Glue Job Status: {status}")
    if status == 'RUNNING':
        print(f"Running for: {exec_time} seconds")
    elif status == 'FAILED':
        error = response['JobRun'].get('ErrorMessage', 'No error message')
        print(f"Error: {error[:200]}")
    
    return status

def main():
    # Check data
    wells, prod = check_tables()
    
    # Check job
    status = check_job_status()
    
    print("\n" + "="*60)
    if wells > 0 and prod > 0:
        print("✅ Data import successful!")
        print("\nNext steps:")
        print("1. Run: python src/data/early_production.py")
        print("2. Run: python src/data/candidate_pool.py")
    elif status == 'RUNNING':
        print("⏳ Job still running, check again later")
    else:
        print("❌ Check job logs for issues")

if __name__ == "__main__":
    main()