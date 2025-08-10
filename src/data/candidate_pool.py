"""
Step 2: Precompute Candidate Pools
Generates analog candidates with spatial/temporal filters
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.db_connector import DatabaseConnector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CandidatePoolBuilder:
    """Builds candidate pools for analog selection"""
    
    def __init__(self, 
                 db_connector: Optional[DatabaseConnector] = None,
                 distance_miles: float = 15.0,
                 lateral_length_tolerance: float = 0.2,
                 max_zero_months: int = 2):
        """
        Initialize candidate pool builder
        
        Args:
            db_connector: Database connection manager
            distance_miles: Maximum distance for candidates (miles)
            lateral_length_tolerance: Tolerance for lateral length matching (±20% default)
            max_zero_months: Maximum allowed zero production months
        """
        self.db = db_connector or DatabaseConnector()
        self.early_rates_table = self.db.tables['early_rates']
        self.candidates_table = self.db.tables['candidates']
        
        # Filter parameters
        self.distance_miles = distance_miles
        self.lateral_length_tolerance = lateral_length_tolerance
        self.max_zero_months = max_zero_months
    
    def create_candidates_table(self):
        """Create the analog_candidates table if it doesn't exist"""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.candidates_table} (
            id SERIAL PRIMARY KEY,
            target_well_id TEXT NOT NULL,
            candidate_well_id TEXT NOT NULL,
            distance_mi NUMERIC,
            length_ratio NUMERIC,
            delta_length NUMERIC,
            formation_match BOOLEAN,
            same_operator BOOLEAN,
            vintage_gap_years NUMERIC,
            ppf_ratio NUMERIC,  -- Proppant per foot ratio
            fpf_ratio NUMERIC,  -- Fluid per foot ratio
            target_first_prod DATE,
            candidate_first_prod DATE,
            target_operator TEXT,
            candidate_operator TEXT,
            target_formation TEXT,
            candidate_formation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Composite unique constraint
            CONSTRAINT unique_target_candidate UNIQUE(target_well_id, candidate_well_id)
        );
        
        -- Create indexes for efficient querying
        CREATE INDEX IF NOT EXISTS idx_candidates_target 
            ON {self.candidates_table}(target_well_id);
        CREATE INDEX IF NOT EXISTS idx_candidates_candidate 
            ON {self.candidates_table}(candidate_well_id);
        CREATE INDEX IF NOT EXISTS idx_candidates_distance 
            ON {self.candidates_table}(distance_mi);
        CREATE INDEX IF NOT EXISTS idx_candidates_formation_match 
            ON {self.candidates_table}(formation_match);
        CREATE INDEX IF NOT EXISTS idx_candidates_same_operator 
            ON {self.candidates_table}(same_operator);
        """
        
        self.db.execute_query(query, fetch=False)
        logger.info(f"Created/verified table: {self.candidates_table}")
    
    def build_candidate_pools(self, batch_size: int = 10, limit: Optional[int] = None):
        """
        Build candidate pools for all target wells
        
        Args:
            batch_size: Number of target wells to process at once
            limit: Optional limit on number of target wells to process
        """
        
        # Get all eligible target wells
        target_query = f"""
        SELECT 
            well_id,
            operator_name,
            formation,
            latitude,
            longitude,
            first_prod_date,
            lateral_length,
            proppant_per_ft,
            fluid_per_ft,
            zero_months_count,
            geom
        FROM {self.early_rates_table}
        WHERE zero_months_count <= {self.max_zero_months}
        ORDER BY first_prod_date DESC, well_id
        """
        
        if limit:
            target_query += f" LIMIT {limit}"
        
        targets = pd.DataFrame(self.db.execute_query(target_query))
        
        if targets.empty:
            logger.warning("No target wells found")
            return
        
        logger.info(f"Processing {len(targets)} target wells")
        
        # Process in batches
        for i in range(0, len(targets), batch_size):
            batch = targets.iloc[i:i+batch_size]
            self._process_target_batch(batch)
            logger.info(f"Processed {min(i+batch_size, len(targets))}/{len(targets)} target wells")
    
    def _process_target_batch(self, target_batch):
        """Process a batch of target wells"""
        
        for _, target in target_batch.iterrows():
            candidates = self._find_candidates_for_target(target)
            
            if not candidates.empty:
                self._insert_candidates(target, candidates)
    
    def _find_candidates_for_target(self, target) -> pd.DataFrame:
        """
        Find candidate analogs for a single target well
        
        Returns filtered candidates with computed features
        """
        
        # Query for candidates with all filters applied
        candidate_query = f"""
        WITH target AS (
            SELECT 
                %(target_well_id)s as well_id,
                %(target_lat)s::numeric as latitude,
                %(target_lon)s::numeric as longitude,
                %(target_first_prod)s::date as first_prod_date,
                %(target_lateral)s::numeric as lateral_length,
                %(target_ppf)s::numeric as proppant_per_ft,
                %(target_fpf)s::numeric as fluid_per_ft,
                %(target_formation)s as formation,
                %(target_operator)s as operator_name
        )
        SELECT 
            c.well_id as candidate_well_id,
            c.operator_name as candidate_operator,
            c.formation as candidate_formation,
            c.first_prod_date as candidate_first_prod,
            c.lateral_length as candidate_lateral,
            c.proppant_per_ft as candidate_ppf,
            c.fluid_per_ft as candidate_fpf,
            
            -- Distance in miles
            ST_Distance(
                ST_MakePoint(t.longitude, t.latitude)::geography,
                c.geom::geography
            ) / 1609.34 as distance_mi,
            
            -- Length ratios
            c.lateral_length / NULLIF(t.lateral_length, 0) as length_ratio,
            c.lateral_length - t.lateral_length as delta_length,
            
            -- Formation and operator matching
            (c.formation = t.formation OR 
             (c.formation IS NULL AND t.formation IS NULL)) as formation_match,
            (c.operator_name = t.operator_name) as same_operator,
            
            -- Vintage gap (years)
            EXTRACT(YEAR FROM t.first_prod_date) - 
            EXTRACT(YEAR FROM c.first_prod_date) as vintage_gap_years,
            
            -- Completion design ratios
            c.proppant_per_ft / NULLIF(t.proppant_per_ft, 0) as ppf_ratio,
            c.fluid_per_ft / NULLIF(t.fluid_per_ft, 0) as fpf_ratio
            
        FROM {self.early_rates_table} c, target t
        WHERE 
            -- Not the same well
            c.well_id != t.well_id
            
            -- No future data leakage
            AND c.first_prod_date <= t.first_prod_date
            
            -- Distance constraint (using spatial index)
            AND ST_DWithin(
                c.geom::geography,
                ST_MakePoint(t.longitude, t.latitude)::geography,
                %(max_distance_m)s  -- distance in meters
            )
            
            -- Lateral length constraint
            AND c.lateral_length BETWEEN 
                t.lateral_length * %(min_length_ratio)s 
                AND t.lateral_length * %(max_length_ratio)s
            
            -- Quality filter
            AND c.zero_months_count <= %(max_zero_months)s
        
        ORDER BY distance_mi, ABS(1 - (c.lateral_length / NULLIF(t.lateral_length, 0)))
        """
        
        params = {
            'target_well_id': target['well_id'],
            'target_lat': float(target['latitude']) if pd.notna(target['latitude']) else 0,
            'target_lon': float(target['longitude']) if pd.notna(target['longitude']) else 0,
            'target_first_prod': target['first_prod_date'],
            'target_lateral': float(target['lateral_length']) if pd.notna(target['lateral_length']) else 0,
            'target_ppf': float(target['proppant_per_ft']) if pd.notna(target['proppant_per_ft']) else 0,
            'target_fpf': float(target['fluid_per_ft']) if pd.notna(target['fluid_per_ft']) else 0,
            'target_formation': target['formation'],
            'target_operator': target['operator_name'],
            'max_distance_m': self.distance_miles * 1609.34,  # Convert miles to meters
            'min_length_ratio': 1 - self.lateral_length_tolerance,
            'max_length_ratio': 1 + self.lateral_length_tolerance,
            'max_zero_months': self.max_zero_months
        }
        
        candidates = pd.DataFrame(self.db.execute_query(candidate_query, params))
        
        return candidates
    
    def _insert_candidates(self, target, candidates):
        """Insert candidate records into the database"""
        
        records = []
        for _, candidate in candidates.iterrows():
            record = {
                'target_well_id': target['well_id'],
                'candidate_well_id': candidate['candidate_well_id'],
                'distance_mi': float(candidate['distance_mi']) if pd.notna(candidate['distance_mi']) else None,
                'length_ratio': float(candidate['length_ratio']) if pd.notna(candidate['length_ratio']) else None,
                'delta_length': float(candidate['delta_length']) if pd.notna(candidate['delta_length']) else None,
                'formation_match': bool(candidate['formation_match']),
                'same_operator': bool(candidate['same_operator']),
                'vintage_gap_years': float(candidate['vintage_gap_years']) if pd.notna(candidate['vintage_gap_years']) else None,
                'ppf_ratio': float(candidate['ppf_ratio']) if pd.notna(candidate['ppf_ratio']) else None,
                'fpf_ratio': float(candidate['fpf_ratio']) if pd.notna(candidate['fpf_ratio']) else None,
                'target_first_prod': target['first_prod_date'],
                'candidate_first_prod': candidate['candidate_first_prod'],
                'target_operator': target['operator_name'],
                'candidate_operator': candidate['candidate_operator'],
                'target_formation': target['formation'],
                'candidate_formation': candidate['candidate_formation']
            }
            records.append(record)
        
        if records:
            insert_query = f"""
            INSERT INTO {self.candidates_table} (
                target_well_id, candidate_well_id, distance_mi,
                length_ratio, delta_length, formation_match,
                same_operator, vintage_gap_years, ppf_ratio, fpf_ratio,
                target_first_prod, candidate_first_prod,
                target_operator, candidate_operator,
                target_formation, candidate_formation
            ) VALUES (
                %(target_well_id)s, %(candidate_well_id)s, %(distance_mi)s,
                %(length_ratio)s, %(delta_length)s, %(formation_match)s,
                %(same_operator)s, %(vintage_gap_years)s, %(ppf_ratio)s, %(fpf_ratio)s,
                %(target_first_prod)s, %(candidate_first_prod)s,
                %(target_operator)s, %(candidate_operator)s,
                %(target_formation)s, %(candidate_formation)s
            )
            ON CONFLICT (target_well_id, candidate_well_id) DO UPDATE SET
                distance_mi = EXCLUDED.distance_mi,
                length_ratio = EXCLUDED.length_ratio,
                created_at = CURRENT_TIMESTAMP
            """
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Use execute_batch for much faster inserts
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, insert_query, records, page_size=100)
                    conn.commit()
    
    def get_statistics(self) -> Dict:
        """Get statistics on the candidate pools"""
        
        stats_query = f"""
        SELECT 
            COUNT(DISTINCT target_well_id) as unique_targets,
            COUNT(DISTINCT candidate_well_id) as unique_candidates,
            COUNT(*) as total_pairs,
            AVG(distance_mi) as avg_distance_mi,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY distance_mi) as median_distance_mi,
            AVG(CASE WHEN formation_match THEN 1 ELSE 0 END) * 100 as pct_formation_match,
            AVG(CASE WHEN same_operator THEN 1 ELSE 0 END) * 100 as pct_same_operator,
            AVG(vintage_gap_years) as avg_vintage_gap,
            MIN(distance_mi) as min_distance_mi,
            MAX(distance_mi) as max_distance_mi
        FROM {self.candidates_table}
        """
        
        stats = self.db.execute_query(stats_query)[0]
        
        # Get distribution of candidates per target
        distribution_query = f"""
        SELECT 
            AVG(candidate_count) as avg_candidates_per_target,
            MIN(candidate_count) as min_candidates,
            MAX(candidate_count) as max_candidates,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY candidate_count) as median_candidates
        FROM (
            SELECT target_well_id, COUNT(*) as candidate_count
            FROM {self.candidates_table}
            GROUP BY target_well_id
        ) t
        """
        
        dist_stats = self.db.execute_query(distribution_query)[0]
        
        logger.info("Candidate Pool Statistics:")
        logger.info(f"  Unique target wells: {stats['unique_targets']:,}")
        logger.info(f"  Unique candidate wells: {stats['unique_candidates']:,}")
        logger.info(f"  Total target-candidate pairs: {stats['total_pairs']:,}")
        logger.info(f"  Avg candidates per target: {dist_stats['avg_candidates_per_target']:.1f}")
        logger.info(f"  Candidates range: {dist_stats['min_candidates']:.0f} - {dist_stats['max_candidates']:.0f}")
        logger.info(f"  Median candidates: {dist_stats['median_candidates']:.0f}")
        logger.info(f"  Avg distance: {stats['avg_distance_mi']:.1f} miles")
        logger.info(f"  Formation match: {stats['pct_formation_match']:.1f}%")
        logger.info(f"  Same operator: {stats['pct_same_operator']:.1f}%")
        logger.info(f"  Avg vintage gap: {stats['avg_vintage_gap']:.1f} years")
        
        return {**stats, **dist_stats}
    
    def get_candidates_for_well(self, well_id: str, max_candidates: int = 100) -> pd.DataFrame:
        """
        Get candidate analogs for a specific well
        
        Args:
            well_id: Target well ID
            max_candidates: Maximum number of candidates to return
        
        Returns:
            DataFrame with candidate information
        """
        
        query = f"""
        SELECT 
            c.*,
            er.avg_oil_m1_9 as candidate_avg_oil,
            er.avg_gas_m1_9 as candidate_avg_gas,
            er.cum_oil_m1_9 as candidate_cum_oil,
            er.cum_gas_m1_9 as candidate_cum_gas
        FROM {self.candidates_table} c
        JOIN {self.early_rates_table} er ON c.candidate_well_id = er.well_id
        WHERE c.target_well_id = %s
        ORDER BY c.distance_mi, ABS(1 - c.length_ratio)
        LIMIT %s
        """
        
        candidates = pd.DataFrame(
            self.db.execute_query(query, (well_id, max_candidates))
        )
        
        return candidates

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build analog candidate pools')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of target wells to process (None for all)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing (default: 10)')
    args = parser.parse_args()
    
    builder = CandidatePoolBuilder(
        distance_miles=15.0,
        lateral_length_tolerance=0.2,
        max_zero_months=2
    )
    
    # Create candidates table
    builder.create_candidates_table()
    
    # Build candidate pools
    logger.info(f"Building candidate pools (limit={args.limit})...")
    builder.build_candidate_pools(batch_size=args.batch_size, limit=args.limit)
    
    # Get statistics
    stats = builder.get_statistics()
    
    # Example: Get candidates for a specific well (would need actual well_id)
    # candidates = builder.get_candidates_for_well('05123456789')

if __name__ == "__main__":
    main()