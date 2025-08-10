# DJ Basin Production Database Mapping for Analog Model

## Overview
This document maps the generic column names from the model_plan.md to the actual database schema for the DJ Basin production forecasting model.

## Database Configuration
- **Host:** minrights-pg.chq86qgigowu.us-east-1.rds.amazonaws.com
- **Database:** dde
- **Primary Tables:** 
  - `data.ed_wells` - Well header information
  - `data.ed_prod` - Monthly production data

## Focus Criteria
- **Basin:** DJ Basin (Denver-Julesburg) 
- **State:** Colorado (state_code='CO' or well_id LIKE '05%')
- **Well Type:** Horizontal wells only (orientation='H')
- **Time Period:** Spud date >= 2010-01-01
- **Core Area:** Focus on core DJ Basin area

## Table Schemas

### data.ed_wells (Well Headers)
Based on the production job config and standard Energy Domain structure:
- **well_id** (text) - Unique well identifier (API number)
- **well_name** (text) - Well name
- **operator_name** (text) - Operating company
- **state_code** (text) - State abbreviation
- **county** (text) - County name
- **basin** (text) - Basin name
- **sub_basin** (text) - Sub-basin name
- **latitude** (numeric) - Surface latitude
- **longitude** (numeric) - Surface longitude
- **spud_date** (date) - Spud date
- **completion_date** (date) - Completion date
- **first_prod_date** (date) - First production date
- **well_type** (text) - Well type
- **orientation** (text) - Well orientation (H/V)
- **formation** (text) - Producing formation
- **lateral_length** (numeric) - Lateral length in feet
- **proppant_used** (numeric) - Total proppant in pounds
- **water_used** (numeric) - Total water/fluid in gallons

### data.ed_prod (Production Data)
- **well_id** (text) - Foreign key to ed_wells
- **prod_date** (date) - Production month
- **prod_month** (bigint) - Month index
- **oil_bbls** (double precision) - Monthly oil production in barrels
- **gas_mcf** (double precision) - Monthly gas production in MCF
- **water_bbls** (double precision) - Monthly water production in barrels
- **cum_oil_bbls** (double precision) - Cumulative oil in barrels
- **cum_gas_mcf** (double precision) - Cumulative gas in MCF
- **created_at** (timestamp) - Record creation timestamp

## Column Mapping for Analog Model

### Model Variable → Database Column Mapping

#### Well Header Mappings (from data.ed_wells)
| Model Variable | Database Column | Calculation/Notes |
|----------------|-----------------|-------------------|
| WELL_ID | well_id | Direct mapping |
| LAT | latitude | Direct mapping |
| LON | longitude | Direct mapping |
| FIRST_PROD_DATE | first_prod_date | Direct mapping |
| FORMATION | formation | Direct mapping |
| OPERATOR | operator_name | Direct mapping |
| LATERAL_LENGTH | lateral_length | In feet |
| PROPPANT_PER_FT | proppant_used / lateral_length | Calculate ratio |
| FLUID_PER_FT | water_used / lateral_length | Calculate ratio |
| SPUD_DATE | spud_date | For filtering |
| COMPLETION_DATE | completion_date | Optional |
| STATE | state_code | Filter = 'CO' |
| BASIN | basin | Filter ILIKE '%DJ%' |
| ORIENTATION | orientation | Filter = 'H' |

#### Production Mappings (from data.ed_prod)
| Model Variable | Database Column | Calculation/Notes |
|----------------|-----------------|-------------------|
| OIL_MO_PROD | oil_bbls | Monthly oil production |
| GAS_MO_PROD | gas_mcf | Monthly gas production |
| WATER_MO_PROD | water_bbls | Monthly water production |
| PROD_DATE | prod_date | Production month |
| PROD_MONTH_INDEX | prod_month | Month counter |
| CUM_OIL | cum_oil_bbls | Cumulative oil |
| CUM_GAS | cum_gas_mcf | Cumulative gas |

## SQL Query Templates

### 1. Build Early Production Table
```sql
WITH dj_wells AS (
    SELECT 
        w.well_id,
        w.operator_name,
        w.formation,
        w.latitude,
        w.longitude,
        w.first_prod_date,
        w.lateral_length,
        w.proppant_used,
        w.water_used,
        w.proppant_used / NULLIF(w.lateral_length, 0) as proppant_per_ft,
        w.water_used / NULLIF(w.lateral_length, 0) as fluid_per_ft
    FROM data.ed_wells w
    WHERE w.state_code = 'CO'
    AND w.orientation = 'H'
    AND w.spud_date >= '2010-01-01'
    AND w.basin ILIKE '%DJ%'
    AND w.first_prod_date IS NOT NULL
),
production_arrays AS (
    SELECT 
        p.well_id,
        ARRAY_AGG(p.oil_bbls ORDER BY p.prod_date) FILTER (
            WHERE p.prod_date >= w.first_prod_date 
            AND p.prod_date < w.first_prod_date + INTERVAL '9 months'
        ) as oil_m1_9,
        ARRAY_AGG(p.gas_mcf ORDER BY p.prod_date) FILTER (
            WHERE p.prod_date >= w.first_prod_date 
            AND p.prod_date < w.first_prod_date + INTERVAL '9 months'
        ) as gas_m1_9
    FROM data.ed_prod p
    JOIN dj_wells w ON p.well_id = w.well_id
    GROUP BY p.well_id
)
SELECT 
    w.*,
    p.oil_m1_9,
    p.gas_m1_9
FROM dj_wells w
JOIN production_arrays p ON w.well_id = p.well_id
WHERE CARDINALITY(p.oil_m1_9) = 9;
```

### 2. Find Candidate Analogs
```sql
-- For a given target well, find candidates within constraints
WITH target AS (
    SELECT * FROM early_rates WHERE well_id = :target_well_id
)
SELECT 
    t.well_id as target_well_id,
    c.well_id as candidate_well_id,
    ST_Distance(
        ST_MakePoint(t.longitude, t.latitude)::geography,
        ST_MakePoint(c.longitude, c.latitude)::geography
    ) / 1609.34 as distance_mi,
    c.lateral_length / NULLIF(t.lateral_length, 0) as length_ratio,
    c.formation = t.formation as formation_match,
    c.operator_name = t.operator_name as same_operator,
    EXTRACT(YEAR FROM t.first_prod_date) - EXTRACT(YEAR FROM c.first_prod_date) as vintage_gap_years
FROM early_rates c, target t
WHERE c.well_id != t.well_id
AND c.first_prod_date <= t.first_prod_date  -- No future leakage
AND ST_DWithin(
    ST_MakePoint(t.longitude, t.latitude)::geography,
    ST_MakePoint(c.longitude, c.latitude)::geography,
    15 * 1609.34  -- 15 miles in meters
)
AND c.lateral_length BETWEEN t.lateral_length * 0.8 AND t.lateral_length * 1.2;
```

## Implementation Notes

1. **Colorado Well Identification**: Colorado API numbers start with '05', so `well_id LIKE '05%'` can be used as an alternative filter

2. **Data Quality Checks**:
   - Filter out wells with NULL first_prod_date
   - Remove wells with >2 zero production months in months 1-9
   - Ensure lateral_length > 0 for ratio calculations

3. **Performance Optimization**:
   - Create indexes on: well_id, state_code, basin, orientation, spud_date, first_prod_date
   - Consider materialized views for the early_rates table
   - Use PostGIS spatial indexes for distance queries

4. **Formation Mapping**:
   - May need to standardize formation names (e.g., "NIOBRARA" vs "Niobrara")
   - Consider grouping similar formations

5. **Production Alignment**:
   - Align production to calendar months from first_prod_date
   - Handle partial month production appropriately

## Next Steps

1. Verify actual column names by connecting to database
2. Create indexes for performance
3. Build early_rates materialized view
4. Implement candidate pool generation
5. Create curve embedding features
6. Train analog selection model