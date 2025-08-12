"""
Fast candidate pool generation using single spatial join
Much more efficient than processing wells one at a time
"""

import logging
import argparse
from src.data.db_connector import DatabaseConnector
from src.config.basin_config import BasinConfig
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_candidate_pool_fast(basin='dj_basin'):
    """Generate candidate pools using efficient spatial join"""
    
    logger.info(f"Starting fast candidate pool generation for {basin}")
    
    db = DatabaseConnector()
    config = BasinConfig(basin)
    
    # Get configuration parameters
    distance_miles = config.get('max_distance_miles', 15.0)
    lateral_tolerance = config.get('lateral_length_tolerance', 0.2)
    min_vintage_year = config.get('min_vintage_year', 2014)
    max_zero_months = config.get('max_zero_production_months', 2)
    
    # Basin-specific API filter
    api_prefixes = config.get('api_prefixes', [])
    if api_prefixes:
        api_filter = " OR ".join([f"t.well_id LIKE '{prefix}%%'" for prefix in api_prefixes])
        api_filter_candidate = " OR ".join([f"c.well_id LIKE '{prefix}%%'" for prefix in api_prefixes])
    else:
        api_filter = "TRUE"
        api_filter_candidate = "TRUE"
    
    # Clear existing candidates for this basin
    clear_query = f"""
    DELETE FROM data.analog_candidates 
    WHERE basin_name = '{config.basin_name}'
    """
    
    # Single efficient spatial join query
    insert_query = f"""
    INSERT INTO data.analog_candidates (
        target_well_id,
        candidate_well_id,
        distance_mi,
        length_ratio,
        delta_length,
        formation_match,
        same_operator,
        vintage_gap_years,
        ppf_ratio,
        fpf_ratio,
        target_first_prod,
        candidate_first_prod,
        target_operator,
        candidate_operator,
        target_formation,
        candidate_formation,
        basin_name
    )
    SELECT 
        t.well_id as target_well_id,
        c.well_id as candidate_well_id,
        
        -- Distance in miles
        ST_Distance(t.geom::geography, c.geom::geography) / 1609.34 as distance_mi,
        
        -- Length ratios
        c.lateral_length / NULLIF(t.lateral_length, 0) as length_ratio,
        c.lateral_length - t.lateral_length as delta_length,
        
        -- Formation and operator matching
        (UPPER(c.formation) = UPPER(t.formation) OR 
         (c.formation IS NULL AND t.formation IS NULL)) as formation_match,
        (c.operator_name = t.operator_name) as same_operator,
        
        -- Vintage gap (years)
        EXTRACT(YEAR FROM t.first_prod_date) - 
        EXTRACT(YEAR FROM c.first_prod_date) as vintage_gap_years,
        
        -- Completion design ratios
        c.proppant_per_ft / NULLIF(t.proppant_per_ft, 0) as ppf_ratio,
        c.fluid_per_ft / NULLIF(t.fluid_per_ft, 0) as fpf_ratio,
        
        -- Store dates
        t.first_prod_date as target_first_prod,
        c.first_prod_date as candidate_first_prod,
        
        -- Store operators
        t.operator_name as target_operator,
        c.operator_name as candidate_operator,
        
        -- Store formations
        t.formation as target_formation,
        c.formation as candidate_formation,
        
        -- Basin
        '{config.basin_name}' as basin_name
        
    FROM data.early_rates t
    INNER JOIN data.early_rates c ON TRUE  -- Cartesian product with filters
    WHERE 
        -- Both wells in same basin
        t.basin_name = '{config.basin_name}'
        AND c.basin_name = '{config.basin_name}'
        
        -- Not the same well
        AND t.well_id != c.well_id
        
        -- Basin API filter for targets
        AND ({api_filter})
        
        -- Basin API filter for candidates  
        AND ({api_filter_candidate})
        
        -- No future data leakage
        AND c.first_prod_date <= t.first_prod_date
        
        -- Vintage filter for candidates
        AND EXTRACT(YEAR FROM c.first_prod_date) >= {min_vintage_year}
        
        -- Distance constraint using spatial index (most important for performance!)
        AND ST_DWithin(
            t.geom::geography,
            c.geom::geography,
            {distance_miles * 1609.34}  -- Convert miles to meters
        )
        
        -- Lateral length constraint
        AND c.lateral_length BETWEEN 
            t.lateral_length * {1 - lateral_tolerance}
            AND t.lateral_length * {1 + lateral_tolerance}
        
        -- Quality filter - skip wells with too many zero months
        AND COALESCE((
            SELECT COUNT(*) 
            FROM unnest(c.oil_m1_9) AS val 
            WHERE val = 0
        ), 0) <= {max_zero_months}
        
        -- Only process target wells from recent vintage
        AND EXTRACT(YEAR FROM t.first_prod_date) >= {min_vintage_year}
    """
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Start timing
        start_time = time.time()
        
        # Clear existing candidates
        logger.info(f"Clearing existing candidates for {basin}...")
        cursor.execute(clear_query)
        deleted = cursor.rowcount
        logger.info(f"Deleted {deleted:,} existing candidates")
        
        # Run the big spatial join
        logger.info(f"Running spatial join for {basin}...")
        logger.info(f"Distance limit: {distance_miles} miles")
        logger.info(f"Lateral tolerance: ±{lateral_tolerance*100:.0f}%")
        
        cursor.execute(insert_query)
        inserted = cursor.rowcount
        
        conn.commit()
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Inserted {inserted:,} candidate pairs in {elapsed:.1f} seconds")
        
        # Get statistics
        stats_query = f"""
        SELECT 
            COUNT(DISTINCT target_well_id) as targets,
            COUNT(DISTINCT candidate_well_id) as candidates,
            COUNT(*) as total_pairs,
            AVG(distance_mi) as avg_distance,
            MAX(distance_mi) as max_distance,
            AVG(CASE WHEN formation_match THEN 1 ELSE 0 END) * 100 as formation_match_pct
        FROM data.analog_candidates
        WHERE basin_name = '{config.basin_name}'
        """
        
        cursor.execute(stats_query)
        stats = cursor.fetchone()
        
        logger.info(f"\nCandidate Pool Statistics for {basin}:")
        logger.info(f"  Target wells: {stats[0]:,}")
        logger.info(f"  Unique candidates: {stats[1]:,}")
        logger.info(f"  Total pairs: {stats[2]:,}")
        logger.info(f"  Avg candidates per target: {stats[2]/stats[0]:.0f}")
        logger.info(f"  Avg distance: {stats[3]:.1f} miles")
        logger.info(f"  Max distance: {stats[4]:.1f} miles")
        logger.info(f"  Formation match: {stats[5]:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Fast candidate pool generation')
    parser.add_argument('--basin', type=str, default='dj_basin',
                        help='Basin to process (dj_basin or bakken)')
    args = parser.parse_args()
    
    generate_candidate_pool_fast(args.basin)

if __name__ == "__main__":
    main()