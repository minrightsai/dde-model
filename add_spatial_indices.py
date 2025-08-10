#!/usr/bin/env python3
"""
Add spatial indices to optimize candidate pool generation
"""

import psycopg2
from src.data.db_connector import DatabaseConnector

def add_spatial_indices():
    """Add spatial and other performance indices"""
    
    db = DatabaseConnector()
    
    indices = [
        # Spatial index on early_rates geometry
        """CREATE INDEX IF NOT EXISTS idx_early_rates_geom 
           ON data.early_rates USING GIST (geom)""",
        
        # Spatial index on model_wells if we add geometry
        """CREATE INDEX IF NOT EXISTS idx_model_wells_geom 
           ON data.model_wells USING GIST (ST_SetSRID(ST_MakePoint(lon, lat), 4326))""",
        
        # B-tree indices for filtering
        """CREATE INDEX IF NOT EXISTS idx_early_rates_first_prod 
           ON data.early_rates (first_prod_date)""",
        
        """CREATE INDEX IF NOT EXISTS idx_early_rates_lateral 
           ON data.early_rates (lateral_length)""",
        
        """CREATE INDEX IF NOT EXISTS idx_early_rates_zero_months 
           ON data.early_rates (zero_months_count)""",
        
        """CREATE INDEX IF NOT EXISTS idx_early_rates_well_id 
           ON data.early_rates (well_id)""",
        
        # Composite index for common query patterns
        """CREATE INDEX IF NOT EXISTS idx_early_rates_composite 
           ON data.early_rates (first_prod_date, lateral_length, zero_months_count)""",
        
        # Indices on analog_candidates table
        """CREATE INDEX IF NOT EXISTS idx_candidates_target 
           ON data.analog_candidates (target_well_id)""",
        
        """CREATE INDEX IF NOT EXISTS idx_candidates_candidate 
           ON data.analog_candidates (candidate_well_id)""",
        
        """CREATE INDEX IF NOT EXISTS idx_candidates_distance 
           ON data.analog_candidates (distance_mi)""",
        
        # Composite index for candidate lookups
        """CREATE INDEX IF NOT EXISTS idx_candidates_lookup 
           ON data.analog_candidates (target_well_id, distance_mi)"""
    ]
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            print("Adding spatial and performance indices...")
            
            for idx, index_sql in enumerate(indices, 1):
                try:
                    print(f"  [{idx}/{len(indices)}] Creating index...")
                    cur.execute(index_sql)
                    conn.commit()
                    print(f"  [{idx}/{len(indices)}] ✓ Success")
                except Exception as e:
                    print(f"  [{idx}/{len(indices)}] ✗ Error: {e}")
                    conn.rollback()
            
            # Analyze tables to update statistics
            print("\nUpdating table statistics...")
            for table in ['data.early_rates', 'data.analog_candidates', 'data.model_wells', 'data.model_prod']:
                try:
                    cur.execute(f"ANALYZE {table}")
                    print(f"  ✓ Analyzed {table}")
                except Exception as e:
                    print(f"  ✗ Error analyzing {table}: {e}")
            
            conn.commit()
    
    print("\n✅ Spatial indices added successfully!")
    print("\nTo check index usage, run:")
    print("  EXPLAIN (ANALYZE, BUFFERS) <your_query>;")

if __name__ == "__main__":
    add_spatial_indices()