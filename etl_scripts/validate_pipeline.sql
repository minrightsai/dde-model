-- ETL Pipeline Validation Queries
-- Run these after the complete pipeline to validate data quality

-- 1. Check basin consistency between model_wells and model_prod
SELECT 
    'Basin Consistency Check' as test_name,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL - ' || COUNT(*) || ' mismatched basins'
    END as result
FROM (
    SELECT w.well_id
    FROM data.model_wells w
    JOIN data.model_prod p ON w.well_id = p.well_id
    WHERE w.basin != p.basin OR w.basin_name != p.basin_name
) mismatches;

-- 2. Verify first_prod_date accuracy
SELECT 
    'First Prod Date Accuracy' as test_name,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL - ' || COUNT(*) || ' wells have incorrect first_prod_date'
    END as result
FROM (
    SELECT w.well_id
    FROM data.model_wells w
    JOIN (
        SELECT well_id, MIN(prod_date) as actual_first_prod
        FROM data.model_prod 
        GROUP BY well_id
    ) p ON w.well_id = p.well_id
    WHERE w.first_prod_date != p.actual_first_prod
) mismatches;

-- 3. Check production month alignment (peak month should be 0)
SELECT 
    'Peak Month Alignment' as test_name,
    'Peak months 0-6: ' || COUNT(*) || ' wells' as result
FROM (
    SELECT well_id, MIN(prod_month) as first_month
    FROM data.model_prod
    GROUP BY well_id
    HAVING MIN(prod_month) BETWEEN 0 AND 6
) aligned_wells;

-- 4. Verify geographic bounds for DJ Basin
SELECT 
    'DJ Basin Geographic Bounds' as test_name,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL - ' || COUNT(*) || ' DJ wells outside bounds'
    END as result
FROM data.model_wells
WHERE basin = 'dj' 
  AND (
    state != 'CO' 
    OR lat NOT BETWEEN 39.51 AND 42.19 
    OR lon NOT BETWEEN -105.13 AND -101.99
  );

-- 5. Verify geographic bounds for Bakken
SELECT 
    'Bakken Geographic Bounds' as test_name,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL - ' || COUNT(*) || ' Bakken wells outside bounds'
    END as result
FROM data.model_wells
WHERE basin = 'bakken' 
  AND (
    state NOT IN ('ND', 'MT')
    OR lat NOT BETWEEN 46.66 AND 48.99 
    OR lon NOT BETWEEN -105.26 AND -102.00
  );

-- 6. Check for wells with insufficient production history
SELECT 
    'Production History Check' as test_name,
    'All wells have 9+ months: ' || 
    CASE 
        WHEN MIN(month_count) >= 9 THEN 'PASS'
        ELSE 'FAIL - Min months: ' || MIN(month_count)
    END as result
FROM (
    SELECT well_id, COUNT(DISTINCT prod_date) as month_count
    FROM data.model_prod
    GROUP BY well_id
) well_months;

-- 7. Data volume summary
SELECT 'Data Volume Summary' as test_name, '' as result;

SELECT 
    basin,
    basin_name,
    COUNT(DISTINCT w.well_id) as wells_total,
    COUNT(DISTINCT p.well_id) as wells_with_production,
    COUNT(p.*) as production_records,
    ROUND(COUNT(p.*) / COUNT(DISTINCT p.well_id), 0) as avg_records_per_well
FROM data.model_wells w
LEFT JOIN data.model_prod p ON w.well_id = p.well_id
GROUP BY basin, basin_name
ORDER BY basin, basin_name;

-- 8. Date range validation
SELECT 'Date Ranges' as test_name, '' as result;

SELECT 
    basin_name,
    MIN(spud_date) as earliest_spud,
    MAX(spud_date) as latest_spud,
    MIN(first_prod_date) as earliest_first_prod,
    MAX(first_prod_date) as latest_first_prod
FROM data.model_wells
WHERE first_prod_date IS NOT NULL
GROUP BY basin_name
ORDER BY basin_name;

SELECT 
    basin_name,
    MIN(prod_date) as earliest_prod_data,
    MAX(prod_date) as latest_prod_data,
    COUNT(DISTINCT DATE_TRUNC('month', prod_date)) as months_of_data
FROM data.model_prod
GROUP BY basin_name
ORDER BY basin_name;

-- 9. Quality metrics by basin
SELECT 'Quality Metrics' as test_name, '' as result;

SELECT 
    basin_name,
    COUNT(*) as total_wells,
    COUNT(lateral_length) as wells_with_lateral_length,
    COUNT(proppant_lbs) as wells_with_proppant,
    COUNT(fluid_gals) as wells_with_fluid,
    ROUND(AVG(lateral_length), 0) as avg_lateral_length,
    ROUND(AVG(proppant_lbs), 0) as avg_proppant_lbs
FROM data.model_wells
GROUP BY basin_name
ORDER BY basin_name;