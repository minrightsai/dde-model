-- Update first_prod_date in model_wells from actual production data
-- This runs after both Wells ETL and Production ETL complete

-- Log before state
SELECT 'Before update:' as status, 
       COUNT(*) as total_wells,
       COUNT(first_prod_date) as wells_with_first_prod_date,
       COUNT(*) - COUNT(first_prod_date) as wells_missing_first_prod_date
FROM data.model_wells;

-- Update first_prod_date from actual production MIN(date)
UPDATE data.model_wells 
SET first_prod_date = (
    SELECT MIN(prod_date) 
    FROM data.model_prod 
    WHERE model_prod.well_id = model_wells.well_id
)
WHERE EXISTS (
    SELECT 1 
    FROM data.model_prod 
    WHERE model_prod.well_id = model_wells.well_id
);

-- Log after state
SELECT 'After update:' as status,
       COUNT(*) as total_wells,
       COUNT(first_prod_date) as wells_with_first_prod_date,
       COUNT(*) - COUNT(first_prod_date) as wells_missing_first_prod_date
FROM data.model_wells;

-- Show wells without production data (these will fall out in downstream joins)
SELECT 'Wells without production:' as status,
       basin,
       basin_name,
       COUNT(*) as wells_without_production
FROM data.model_wells w
WHERE NOT EXISTS (
    SELECT 1 FROM data.model_prod p WHERE p.well_id = w.well_id
)
GROUP BY basin, basin_name
ORDER BY basin, basin_name;

-- Validate: Check for wells where first_prod_date doesn't match actual MIN(prod_date)
-- This should return 0 rows
SELECT 'Data quality check:' as status,
       COUNT(*) as mismatched_dates
FROM data.model_wells w
JOIN data.model_prod p ON w.well_id = p.well_id
WHERE w.first_prod_date IS NOT NULL
GROUP BY w.well_id, w.first_prod_date
HAVING w.first_prod_date != MIN(p.prod_date);

-- Summary statistics by basin
SELECT 'Final summary by basin:' as status,
       w.basin,
       w.basin_name,
       COUNT(DISTINCT w.well_id) as total_wells,
       COUNT(DISTINCT p.well_id) as wells_with_production,
       COUNT(DISTINCT w.well_id) - COUNT(DISTINCT p.well_id) as wells_without_production,
       ROUND(
           100.0 * COUNT(DISTINCT p.well_id) / COUNT(DISTINCT w.well_id), 2
       ) as production_percentage
FROM data.model_wells w
LEFT JOIN data.model_prod p ON w.well_id = p.well_id
GROUP BY w.basin, w.basin_name
ORDER BY w.basin, w.basin_name;