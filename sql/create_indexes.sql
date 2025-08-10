-- Database Indexes for DJ Basin Production Model
-- Run these to improve query performance on large tables

-- ============================================
-- INDEXES FOR data.ed_wells TABLE
-- ============================================

-- Primary key index (if not exists)
CREATE INDEX IF NOT EXISTS idx_ed_wells_well_id 
ON data.ed_wells(well_id);

-- State filtering (for Colorado wells)
CREATE INDEX IF NOT EXISTS idx_ed_wells_state 
ON data.ed_wells(state);

-- Well orientation/type filtering
CREATE INDEX IF NOT EXISTS idx_ed_wells_drill_type 
ON data.ed_wells(drill_type);

-- Basin filtering (for DJ Basin)
CREATE INDEX IF NOT EXISTS idx_ed_wells_basin 
ON data.ed_wells(basin);

-- Date filtering
CREATE INDEX IF NOT EXISTS idx_ed_wells_spud_date 
ON data.ed_wells(spud_date);

CREATE INDEX IF NOT EXISTS idx_ed_wells_first_prod_date 
ON data.ed_wells(first_prod_date);

-- Composite index for DJ Basin filtering
CREATE INDEX IF NOT EXISTS idx_ed_wells_dj_basin_composite 
ON data.ed_wells(state, drill_type, spud_date, basin)
WHERE state = 'CO' 
  AND drill_type = 'Horizontal' 
  AND spud_date >= '2010-01-01';

-- Spatial index for location-based queries (if PostGIS is enabled)
-- First create geometry column if needed:
-- ALTER TABLE data.ed_wells ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
-- UPDATE data.ed_wells SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326) WHERE lon IS NOT NULL AND lat IS NOT NULL;
-- CREATE INDEX IF NOT EXISTS idx_ed_wells_geom ON data.ed_wells USING GIST(geom);

-- Colorado well prefix index
CREATE INDEX IF NOT EXISTS idx_ed_wells_co_prefix 
ON data.ed_wells(well_id)
WHERE well_id LIKE '05%';

-- ============================================
-- INDEXES FOR data.ed_prod TABLE  
-- ============================================

-- Well ID for joins
CREATE INDEX IF NOT EXISTS idx_ed_prod_well_id 
ON data.ed_prod(well_id);

-- Production date for time-based queries
CREATE INDEX IF NOT EXISTS idx_ed_prod_prod_date 
ON data.ed_prod(prod_date);

-- Composite index for well + date lookups
CREATE INDEX IF NOT EXISTS idx_ed_prod_well_date 
ON data.ed_prod(well_id, prod_date);

-- Index for Colorado wells production
CREATE INDEX IF NOT EXISTS idx_ed_prod_co_wells 
ON data.ed_prod(well_id)
WHERE well_id LIKE '05%';

-- ============================================
-- ANALYZE TABLES AFTER INDEX CREATION
-- ============================================
ANALYZE data.ed_wells;
ANALYZE data.ed_prod;

-- ============================================
-- OPTIONAL: Materialized View for DJ Basin subset
-- ============================================

-- Create materialized view for faster access to DJ Basin wells
CREATE MATERIALIZED VIEW IF NOT EXISTS data.dj_basin_wells AS
SELECT 
    w.*,
    ST_SetSRID(ST_MakePoint(w.lon, w.lat), 4326) as geom
FROM data.ed_wells w
WHERE w.state = 'CO'
  AND w.drill_type = 'Horizontal'
  AND w.spud_date >= '2010-01-01'
  AND (w.basin ILIKE '%DJ%' OR w.well_id LIKE '05%')
  AND w.lateral_length > 0
  AND w.first_prod_date IS NOT NULL;

-- Index the materialized view
CREATE INDEX IF NOT EXISTS idx_dj_basin_wells_well_id 
ON data.dj_basin_wells(well_id);

CREATE INDEX IF NOT EXISTS idx_dj_basin_wells_first_prod 
ON data.dj_basin_wells(first_prod_date);

CREATE INDEX IF NOT EXISTS idx_dj_basin_wells_geom 
ON data.dj_basin_wells USING GIST(geom);

-- ============================================
-- Query to check index usage
-- ============================================
/*
To check if indexes are being used:

EXPLAIN (ANALYZE, BUFFERS) 
SELECT COUNT(*) 
FROM data.ed_wells 
WHERE state = 'CO' 
  AND drill_type = 'Horizontal' 
  AND spud_date >= '2010-01-01';
*/