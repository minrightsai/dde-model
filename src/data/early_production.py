"""
Step 1: Build Early Production Table
Creates a table with first 9 months of production for each well
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.db_connector import DatabaseConnector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EarlyProductionBuilder:
    """Builds early production table with 9-month production arrays"""
    
    def __init__(self, db_connector=None):
        """Initialize with database connector"""
        self.db = db_connector or DatabaseConnector()
        self.wells_table = self.db.tables['wells']
        self.prod_table = self.db.tables['production']
        self.early_rates_table = self.db.tables['early_rates']
    
    def create_early_rates_table(self):
        """Create the early_rates table if it doesn't exist"""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.early_rates_table} (
            well_id TEXT PRIMARY KEY,
            operator_name TEXT,
            formation TEXT,
            latitude NUMERIC,
            longitude NUMERIC,
            first_prod_date DATE,
            lateral_length NUMERIC,
            proppant_used NUMERIC,
            water_used NUMERIC,
            proppant_per_ft NUMERIC,
            fluid_per_ft NUMERIC,
            oil_m1_9 NUMERIC[],
            gas_m1_9 NUMERIC[],
            oil_m1_9_norm NUMERIC[],  -- Normalized per 1000 ft
            gas_m1_9_norm NUMERIC[],
            oil_m1_9_log NUMERIC[],   -- log1p transformed
            gas_m1_9_log NUMERIC[],
            zero_months_count INTEGER,
            avg_oil_m1_9 NUMERIC,
            avg_gas_m1_9 NUMERIC,
            cum_oil_m1_9 NUMERIC,
            cum_gas_m1_9 NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            geom GEOMETRY(Point, 4326)  -- PostGIS spatial column
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_early_rates_first_prod_date 
            ON {self.early_rates_table}(first_prod_date);
        CREATE INDEX IF NOT EXISTS idx_early_rates_formation 
            ON {self.early_rates_table}(formation);
        CREATE INDEX IF NOT EXISTS idx_early_rates_operator 
            ON {self.early_rates_table}(operator_name);
        CREATE INDEX IF NOT EXISTS idx_early_rates_lateral_length 
            ON {self.early_rates_table}(lateral_length);
        CREATE INDEX IF NOT EXISTS idx_early_rates_geom 
            ON {self.early_rates_table} USING GIST(geom);
        """
        
        self.db.execute_query(query, fetch=False)
        logger.info(f"Created/verified table: {self.early_rates_table}")
    
    def build_early_production(self, batch_size=100, limit=None):
        """Build early production table in batches
        
        Args:
            batch_size: Number of wells to process in each batch
            limit: Optional limit on total wells to process (for testing)
        """
        
        # Get eligible wells
        wells_query = f"""
        SELECT 
            w.{self.db.column_mapping['WELL_ID']} as well_id,
            w.{self.db.column_mapping['OPERATOR']} as operator_name,
            w.{self.db.column_mapping['FORMATION']} as formation,
            w.{self.db.column_mapping['LAT']} as latitude,
            w.{self.db.column_mapping['LON']} as longitude,
            w.{self.db.column_mapping['FIRST_PROD_DATE']} as first_prod_date,
            w.{self.db.column_mapping['LATERAL_LENGTH']} as lateral_length,
            w.{self.db.column_mapping['PROPPANT_USED']} as proppant_used,
            w.{self.db.column_mapping['WATER_USED']} as water_used
        FROM {self.wells_table} w
        WHERE {self.db.get_dj_basin_filter()}
        AND w.{self.db.column_mapping['FIRST_PROD_DATE']} IS NOT NULL
        AND w.{self.db.column_mapping['LATERAL_LENGTH']} > 0
        ORDER BY w.{self.db.column_mapping['FIRST_PROD_DATE']} DESC, w.{self.db.column_mapping['WELL_ID']}
        """
        
        if limit:
            wells_query += f" LIMIT {limit}"
        
        wells = pd.DataFrame(self.db.execute_query(wells_query))
        
        if wells.empty:
            logger.warning("No wells found matching criteria")
            return
        
        logger.info(f"Processing {len(wells)} wells")
        
        # Process in batches
        for i in range(0, len(wells), batch_size):
            batch = wells.iloc[i:i+batch_size]
            self._process_batch(batch)
            logger.info(f"Processed {min(i+batch_size, len(wells))}/{len(wells)} wells")
    
    def _process_batch(self, wells_batch):
        """Process a batch of wells"""
        
        well_ids = wells_batch['well_id'].tolist()
        
        # Get production data for these wells
        prod_query = f"""
        SELECT 
            p.{self.db.column_mapping['WELL_ID']} as well_id,
            p.{self.db.column_mapping['PROD_DATE']} as prod_date,
            p.{self.db.column_mapping['OIL_MO_PROD']} as oil,
            p.{self.db.column_mapping['GAS_MO_PROD']} as gas
        FROM {self.prod_table} p
        WHERE p.{self.db.column_mapping['WELL_ID']} = ANY(%s)
        ORDER BY p.{self.db.column_mapping['WELL_ID']}, p.{self.db.column_mapping['PROD_DATE']}
        """
        
        production = pd.DataFrame(self.db.execute_query(prod_query, (well_ids,)))
        
        if production.empty:
            logger.warning(f"No production data for batch")
            return
        
        # Process each well
        records = []
        for _, well in wells_batch.iterrows():
            well_prod = production[production['well_id'] == well['well_id']]
            
            if well_prod.empty:
                continue
            
            # Align production to first 9 months
            first_prod_date = pd.to_datetime(well['first_prod_date'])
            well_prod['prod_date'] = pd.to_datetime(well_prod['prod_date'])
            
            # Get first 9 months
            end_date = first_prod_date + pd.DateOffset(months=9)
            early_prod = well_prod[
                (well_prod['prod_date'] >= first_prod_date) & 
                (well_prod['prod_date'] < end_date)
            ].sort_values('prod_date')
            
            if len(early_prod) < 9:
                continue  # Skip wells without full 9 months
            
            # Take first 9 months
            early_prod = early_prod.head(9)
            
            # Extract production arrays
            oil_m1_9 = early_prod['oil'].fillna(0).values[:9]
            gas_m1_9 = early_prod['gas'].fillna(0).values[:9]
            
            # Count zero months
            zero_months = np.sum(oil_m1_9 == 0)
            
            # Skip wells with too many zero months
            if zero_months > 2:
                continue
            
            # Calculate derived features
            lateral_length = well['lateral_length'] if well['lateral_length'] > 0 else 1
            proppant_per_ft = well['proppant_used'] / lateral_length if well['proppant_used'] else 0
            fluid_per_ft = well['water_used'] / lateral_length if well['water_used'] else 0
            
            # Normalized production (per 1000 ft)
            oil_m1_9_norm = oil_m1_9 * 1000 / lateral_length
            gas_m1_9_norm = gas_m1_9 * 1000 / lateral_length
            
            # Log transformed
            oil_m1_9_log = np.log1p(oil_m1_9)
            gas_m1_9_log = np.log1p(gas_m1_9)
            
            # Aggregates
            avg_oil = np.mean(oil_m1_9)
            avg_gas = np.mean(gas_m1_9)
            cum_oil = np.sum(oil_m1_9)
            cum_gas = np.sum(gas_m1_9)
            
            record = {
                'well_id': well['well_id'],
                'operator_name': well['operator_name'],
                'formation': well['formation'],
                'latitude': well['latitude'],
                'longitude': well['longitude'],
                'first_prod_date': well['first_prod_date'],
                'lateral_length': lateral_length,
                'proppant_used': well['proppant_used'],
                'water_used': well['water_used'],
                'proppant_per_ft': proppant_per_ft,
                'fluid_per_ft': fluid_per_ft,
                'oil_m1_9': [float(x) for x in oil_m1_9],
                'gas_m1_9': [float(x) for x in gas_m1_9],
                'oil_m1_9_norm': [float(x) for x in oil_m1_9_norm],
                'gas_m1_9_norm': [float(x) for x in gas_m1_9_norm],
                'oil_m1_9_log': [float(x) for x in oil_m1_9_log],
                'gas_m1_9_log': [float(x) for x in gas_m1_9_log],
                'zero_months_count': int(zero_months),
                'avg_oil_m1_9': float(avg_oil),
                'avg_gas_m1_9': float(avg_gas),
                'cum_oil_m1_9': float(cum_oil),
                'cum_gas_m1_9': float(cum_gas)
            }
            
            records.append(record)
        
        # Insert records
        if records:
            self._insert_records(records)
    
    def _insert_records(self, records):
        """Insert records into early_rates table"""
        
        insert_query = f"""
        INSERT INTO {self.early_rates_table} (
            well_id, operator_name, formation, latitude, longitude,
            first_prod_date, lateral_length, proppant_used, water_used,
            proppant_per_ft, fluid_per_ft, oil_m1_9, gas_m1_9,
            oil_m1_9_norm, gas_m1_9_norm, oil_m1_9_log, gas_m1_9_log,
            zero_months_count, avg_oil_m1_9, avg_gas_m1_9,
            cum_oil_m1_9, cum_gas_m1_9, geom
        ) VALUES (
            %(well_id)s, %(operator_name)s, %(formation)s, %(latitude)s, %(longitude)s,
            %(first_prod_date)s, %(lateral_length)s, %(proppant_used)s, %(water_used)s,
            %(proppant_per_ft)s, %(fluid_per_ft)s, %(oil_m1_9)s, %(gas_m1_9)s,
            %(oil_m1_9_norm)s, %(gas_m1_9_norm)s, %(oil_m1_9_log)s, %(gas_m1_9_log)s,
            %(zero_months_count)s, %(avg_oil_m1_9)s, %(avg_gas_m1_9)s,
            %(cum_oil_m1_9)s, %(cum_gas_m1_9)s,
            ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326)
        )
        ON CONFLICT (well_id) DO UPDATE SET
            oil_m1_9 = EXCLUDED.oil_m1_9,
            gas_m1_9 = EXCLUDED.gas_m1_9,
            oil_m1_9_norm = EXCLUDED.oil_m1_9_norm,
            gas_m1_9_norm = EXCLUDED.gas_m1_9_norm,
            oil_m1_9_log = EXCLUDED.oil_m1_9_log,
            gas_m1_9_log = EXCLUDED.gas_m1_9_log,
            zero_months_count = EXCLUDED.zero_months_count,
            avg_oil_m1_9 = EXCLUDED.avg_oil_m1_9,
            avg_gas_m1_9 = EXCLUDED.avg_gas_m1_9,
            cum_oil_m1_9 = EXCLUDED.cum_oil_m1_9,
            cum_gas_m1_9 = EXCLUDED.cum_gas_m1_9,
            created_at = CURRENT_TIMESTAMP
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                for record in records:
                    cur.execute(insert_query, record)
                conn.commit()
    
    def get_statistics(self):
        """Get statistics on the early_rates table"""
        
        query = f"""
        SELECT 
            COUNT(*) as total_wells,
            COUNT(DISTINCT operator_name) as unique_operators,
            COUNT(DISTINCT formation) as unique_formations,
            AVG(lateral_length) as avg_lateral_length,
            AVG(cum_oil_m1_9) as avg_cum_oil,
            AVG(cum_gas_m1_9) as avg_cum_gas,
            MIN(first_prod_date) as earliest_prod,
            MAX(first_prod_date) as latest_prod
        FROM {self.early_rates_table}
        """
        
        stats = self.db.execute_query(query)[0]
        
        logger.info("Early Production Table Statistics:")
        logger.info(f"  Total wells: {stats['total_wells']:,}")
        logger.info(f"  Unique operators: {stats['unique_operators']}")
        logger.info(f"  Unique formations: {stats['unique_formations']}")
        logger.info(f"  Avg lateral length: {stats['avg_lateral_length']:.0f} ft")
        logger.info(f"  Avg cumulative oil (m1-9): {stats['avg_cum_oil']:.0f} bbl")
        logger.info(f"  Avg cumulative gas (m1-9): {stats['avg_cum_gas']:.0f} mcf")
        logger.info(f"  Date range: {stats['earliest_prod']} to {stats['latest_prod']}")
        
        return stats

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build early production table')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of wells to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of wells to process per batch')
    args = parser.parse_args()
    
    if args.limit:
        logger.info(f"TEST MODE: Processing only {args.limit} wells")
    
    builder = EarlyProductionBuilder()
    
    # Check if tables exist
    if not builder.db.check_tables_exist():
        logger.error("Required tables not found")
        return
    
    # Create early_rates table
    builder.create_early_rates_table()
    
    # Build early production data
    builder.build_early_production(batch_size=args.batch_size, limit=args.limit)
    
    # Get statistics
    builder.get_statistics()

if __name__ == "__main__":
    main()