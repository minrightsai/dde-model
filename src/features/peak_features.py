"""
Step 3: Build Peak Production Features
Generate interpretable peak production features to replace PCA embeddings
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.db_connector import DatabaseConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PeakFeatures:
    """Generate and store peak production features"""
    
    def __init__(self):
        """Initialize peak features generator"""
        self.db = DatabaseConnector()
        
    def load_production_data(self) -> pd.DataFrame:
        """Load early production data from database"""
        query = """
        SELECT 
            well_id,
            basin_name,
            lateral_length,
            proppant_per_ft,
            oil_m1_9,
            gas_m1_9
        FROM data.early_rates
        WHERE oil_m1_9 IS NOT NULL 
        AND gas_m1_9 IS NOT NULL
        AND lateral_length IS NOT NULL
        AND proppant_per_ft IS NOT NULL
        ORDER BY well_id
        """
        
        logger.info("Loading production data...")
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} wells with production data")
        
        return df
    
    def calculate_peak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate peak production features for each well
        
        Args:
            df: DataFrame with production arrays and well characteristics
            
        Returns:
            DataFrame with calculated peak features
        """
        logger.info("Calculating peak production features...")
        
        features_list = []
        
        for _, row in df.iterrows():
            well_id = row['well_id']
            basin_name = row['basin_name']
            lateral_length = float(row['lateral_length'])
            proppant_per_ft = float(row['proppant_per_ft'])
            
            # Convert arrays to numpy arrays (handle Decimal type)
            oil_array = np.array([float(val) for val in row['oil_m1_9']])
            gas_array = np.array([float(val) for val in row['gas_m1_9']])
            
            # Calculate peak values
            peak_oil = np.max(oil_array)
            peak_gas = np.max(gas_array)
            
            # Find month when peak occurred (1-indexed)
            peak_oil_month = np.argmax(oil_array) + 1
            peak_gas_month = np.argmax(gas_array) + 1
            
            # Normalized peaks (per 1000 ft lateral)
            peak_oil_per_kft = peak_oil / (lateral_length / 1000) if lateral_length > 0 else 0
            peak_gas_per_kft = peak_gas / (lateral_length / 1000) if lateral_length > 0 else 0
            
            # Peak efficiency (peak per proppant loading)
            peak_oil_per_ppf = peak_oil / (proppant_per_ft + 1) if proppant_per_ft >= 0 else 0
            
            # Decline rate from peak to month 9
            month_9_oil = oil_array[8]  # 0-indexed, so month 9 is index 8
            month_9_gas = gas_array[8]
            
            oil_decline_rate = (peak_oil - month_9_oil) / (peak_oil + 1) if peak_oil > 0 else 0
            gas_decline_rate = (peak_gas - month_9_gas) / (peak_gas + 1) if peak_gas > 0 else 0
            
            # Ensure decline rates are between 0 and 1
            oil_decline_rate = max(0, min(1, oil_decline_rate))
            gas_decline_rate = max(0, min(1, gas_decline_rate))
            
            features_list.append({
                'well_id': well_id,
                'basin_name': basin_name,
                'peak_oil': peak_oil,
                'peak_gas': peak_gas,
                'peak_oil_month': peak_oil_month,
                'peak_gas_month': peak_gas_month,
                'peak_oil_per_kft': peak_oil_per_kft,
                'peak_gas_per_kft': peak_gas_per_kft,
                'peak_oil_per_ppf': peak_oil_per_ppf,
                'oil_decline_rate': oil_decline_rate,
                'gas_decline_rate': gas_decline_rate
            })
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Calculated peak features for {len(features_df)} wells")
        
        return features_df
    
    def create_peak_features_table(self):
        """Create peak_features table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS data.peak_features (
            well_id TEXT PRIMARY KEY,
            basin_name TEXT,
            peak_oil NUMERIC,
            peak_gas NUMERIC,
            peak_oil_month INTEGER,
            peak_gas_month INTEGER,
            peak_oil_per_kft NUMERIC,
            peak_gas_per_kft NUMERIC,
            peak_oil_per_ppf NUMERIC,
            oil_decline_rate NUMERIC,
            gas_decline_rate NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_peak_features_basin 
        ON data.peak_features(basin_name);
        
        CREATE INDEX IF NOT EXISTS idx_peak_features_peak_oil 
        ON data.peak_features(peak_oil);
        
        CREATE INDEX IF NOT EXISTS idx_peak_features_well 
        ON data.peak_features(well_id);
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
        
        logger.info("Created peak_features table")
    
    def store_peak_features(self, df: pd.DataFrame):
        """Store peak features in database"""
        from psycopg2.extras import execute_batch
        
        insert_query = """
        INSERT INTO data.peak_features (
            well_id, basin_name, peak_oil, peak_gas, 
            peak_oil_month, peak_gas_month, peak_oil_per_kft, 
            peak_gas_per_kft, peak_oil_per_ppf, oil_decline_rate, gas_decline_rate
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (well_id) DO UPDATE SET
            basin_name = EXCLUDED.basin_name,
            peak_oil = EXCLUDED.peak_oil,
            peak_gas = EXCLUDED.peak_gas,
            peak_oil_month = EXCLUDED.peak_oil_month,
            peak_gas_month = EXCLUDED.peak_gas_month,
            peak_oil_per_kft = EXCLUDED.peak_oil_per_kft,
            peak_gas_per_kft = EXCLUDED.peak_gas_per_kft,
            peak_oil_per_ppf = EXCLUDED.peak_oil_per_ppf,
            oil_decline_rate = EXCLUDED.oil_decline_rate,
            gas_decline_rate = EXCLUDED.gas_decline_rate,
            created_at = CURRENT_TIMESTAMP
        """
        
        records = []
        for _, row in df.iterrows():
            records.append((
                row['well_id'],
                row['basin_name'],
                float(row['peak_oil']),
                float(row['peak_gas']),
                int(row['peak_oil_month']),
                int(row['peak_gas_month']),
                float(row['peak_oil_per_kft']),
                float(row['peak_gas_per_kft']),
                float(row['peak_oil_per_ppf']),
                float(row['oil_decline_rate']),
                float(row['gas_decline_rate'])
            ))
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                execute_batch(cursor, insert_query, records, page_size=100)
                conn.commit()
        
        logger.info(f"Stored {len(records)} peak features in database")
    
    def run(self):
        """Main execution method"""
        logger.info("Starting peak features generation...")
        
        # Load production data
        df = self.load_production_data()
        
        if len(df) == 0:
            logger.warning("No production data found")
            return
        
        # Calculate peak features
        features_df = self.calculate_peak_features(df)
        
        # Create table and store features
        self.create_peak_features_table()
        self.store_peak_features(features_df)
        
        logger.info("✅ Peak features generation complete!")
        logger.info(f"Generated peak features for {len(features_df)} wells")
        
        self.print_summary_stats(features_df)
    
    def print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics of peak features"""
        print("\n" + "="*60)
        print("PEAK FEATURES SUMMARY STATISTICS")
        print("="*60)
        
        # Basin breakdown
        basin_counts = df['basin_name'].value_counts()
        print(f"\nWells by Basin:")
        for basin, count in basin_counts.items():
            print(f"  {basin}: {count:,} wells")
        
        # Peak oil statistics
        print(f"\nPeak Oil Production:")
        peak_oil = df['peak_oil']
        print(f"  Mean: {peak_oil.mean():.0f} bbl/month")
        print(f"  Median: {peak_oil.median():.0f} bbl/month")
        print(f"  Min: {peak_oil.min():.0f} bbl/month")
        print(f"  Max: {peak_oil.max():.0f} bbl/month")
        print(f"  Std: {peak_oil.std():.0f} bbl/month")
        
        # Peak gas statistics
        print(f"\nPeak Gas Production:")
        peak_gas = df['peak_gas']
        print(f"  Mean: {peak_gas.mean():.0f} mcf/month")
        print(f"  Median: {peak_gas.median():.0f} mcf/month")
        print(f"  Min: {peak_gas.min():.0f} mcf/month")
        print(f"  Max: {peak_gas.max():.0f} mcf/month")
        print(f"  Std: {peak_gas.std():.0f} mcf/month")
        
        # Peak timing
        print(f"\nPeak Timing:")
        oil_month_counts = df['peak_oil_month'].value_counts().sort_index()
        print(f"  Oil peak months: {dict(oil_month_counts)}")
        gas_month_counts = df['peak_gas_month'].value_counts().sort_index()
        print(f"  Gas peak months: {dict(gas_month_counts)}")
        
        # Normalized production
        print(f"\nNormalized Peak Production (per 1000 ft):")
        print(f"  Oil per kft - Mean: {df['peak_oil_per_kft'].mean():.0f}, Median: {df['peak_oil_per_kft'].median():.0f}")
        print(f"  Gas per kft - Mean: {df['peak_gas_per_kft'].mean():.0f}, Median: {df['peak_gas_per_kft'].median():.0f}")
        
        # Decline rates
        print(f"\nDecline Rates (peak to month 9):")
        print(f"  Oil decline - Mean: {df['oil_decline_rate'].mean():.3f}, Median: {df['oil_decline_rate'].median():.3f}")
        print(f"  Gas decline - Mean: {df['gas_decline_rate'].mean():.3f}, Median: {df['gas_decline_rate'].median():.3f}")
        
        # Early peak wells (months 1-3)
        early_oil_peaks = (df['peak_oil_month'] <= 3).sum()
        early_gas_peaks = (df['peak_gas_month'] <= 3).sum()
        print(f"\nEarly Peak Wells (months 1-3):")
        print(f"  Oil: {early_oil_peaks:,} wells ({early_oil_peaks/len(df)*100:.1f}%)")
        print(f"  Gas: {early_gas_peaks:,} wells ({early_gas_peaks/len(df)*100:.1f}%)")
        
        print("="*60)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate peak production features')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    peak_features = PeakFeatures()
    peak_features.run()


if __name__ == "__main__":
    main()