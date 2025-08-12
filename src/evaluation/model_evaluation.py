"""
Model evaluation framework for comparing Baseline vs LightGBM models
Supports multi-basin evaluation (DJ Basin and Bakken)
Properly handles out-of-sample testing without data leakage
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import logging
from src.data.db_connector import DatabaseConnector
from src.models.baseline import BaselinePicker
from src.evaluation.metrics import calculate_weighted_mae
from src.features.ranking_features import build_ranking_features, get_feature_names
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_wells(db, basin='dj_basin', year=2024):
    """Load test wells from model_wells and their actual production from model_prod"""
    
    # Get wells that started production in the test year with their parameters
    query = """
    WITH test_wells AS (
        -- Find wells that started production in the test year
        SELECT 
            mp.well_id,
            MIN(mp.prod_date) as first_prod_date
        FROM data.model_prod mp
        JOIN data.model_wells mw ON mp.well_id = mw.well_id
        WHERE mw.basin = %s
        GROUP BY mp.well_id
        HAVING EXTRACT(YEAR FROM MIN(mp.prod_date)) = %s
        AND EXTRACT(MONTH FROM MIN(mp.prod_date)) <= 6  -- H1 only
    ),
    production_arrays AS (
        -- Get first 9 months of production for these wells
        SELECT 
            tw.well_id,
            mp.prod_month,
            mp.oil_bbls,
            mp.gas_mcf
        FROM test_wells tw
        JOIN data.model_prod mp ON tw.well_id = mp.well_id
        WHERE mp.prod_month < 9  -- First 9 months only
    ),
    aggregated_prod AS (
        -- Create production arrays
        SELECT 
            well_id,
            array_agg(COALESCE(oil_bbls, 0) ORDER BY prod_month) as oil_m1_9,
            array_agg(COALESCE(gas_mcf, 0) ORDER BY prod_month) as gas_m1_9,
            COUNT(*) as num_months
        FROM production_arrays
        GROUP BY well_id
        HAVING COUNT(*) = 9  -- Only wells with full 9 months
    )
    -- Join with well metadata
    SELECT 
        mw.well_id,
        mw.operator as operator_name,
        mw.formation,
        mw.lat,
        mw.lon,
        mw.lateral_length,
        CASE 
            WHEN mw.lateral_length > 0 THEN mw.proppant_lbs / mw.lateral_length
            ELSE 0
        END as proppant_per_ft,
        CASE 
            WHEN mw.lateral_length > 0 THEN mw.fluid_gals / mw.lateral_length
            ELSE 0
        END as fluid_per_ft,
        tw.first_prod_date,
        CASE 
            WHEN mw.basin = 'dj' THEN 'dj_basin'
            WHEN mw.basin = 'bakken' THEN 'bakken'
            ELSE mw.basin
        END as basin_name,
        ap.oil_m1_9,
        ap.gas_m1_9
    FROM test_wells tw
    JOIN data.model_wells mw ON tw.well_id = mw.well_id
    JOIN aggregated_prod ap ON tw.well_id = ap.well_id
    WHERE mw.formation IS NOT NULL
    AND mw.lateral_length > 0
    AND mw.lat IS NOT NULL
    AND mw.lon IS NOT NULL
    ORDER BY tw.first_prod_date DESC
    """
    
    with db.get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(
            'dj' if basin == 'dj_basin' else basin,
            year
        ))
    
    logger.info(f"Loaded {len(df)} test wells from {year} H1 for {basin}")
    return df

def evaluate_baseline(test_wells, basin='dj_basin'):
    """Evaluate baseline model on test wells
    
    Note: The baseline model's predict method doesn't apply warping,
    but for fair comparison with LightGBM, we should apply the same warping
    that was used during training.
    """
    
    picker = BaselinePicker(basin=basin)
    errors = []
    predictions = []
    successful = 0
    failed = 0
    no_analogs = 0
    
    logger.info(f"Evaluating Baseline model for {basin}...")
    logger.info("Note: Baseline predict method doesn't apply warping - predictions may not match training")
    
    for idx, well in test_wells.iterrows():
        try:
            # Use the baseline's predict method which handles everything
            result = picker.predict(well.to_dict())
            
            if 'error' in result:
                if 'No candidates' in result.get('error', ''):
                    no_analogs += 1
                failed += 1
                logger.debug(f"Well {well['well_id']}: {result.get('error')}")
                continue
            
            # Extract prediction
            predicted = np.array(result.get('p50_oil_m1_9', []))
            
            if len(predicted) == 0 or len(predicted) != 9:
                failed += 1
                continue
                
            # Calculate error (convert to float to handle Decimal types from PostgreSQL)
            actual = np.array(well['oil_m1_9'], dtype=float)
            error = calculate_weighted_mae(actual, predicted)
            
            errors.append(error)
            predictions.append({
                'well_id': well['well_id'],
                'formation': well['formation'],
                'actual': actual,
                'predicted': predicted,
                'error': error,
                'n_analogs': result.get('n_analogs', 0)
            })
            successful += 1
            
            if successful % 10 == 0:
                logger.info(f"Processed {successful} wells, median error: {np.median(errors):.0f} bbls")
                
        except Exception as e:
            logger.error(f"Error processing well {well['well_id']}: {e}")
            failed += 1
            continue
    
    logger.info(f"Baseline complete: {successful} successful, {failed} failed ({no_analogs} had no analogs)")
    return errors, predictions

def evaluate_lightgbm_simple(test_wells, basin='dj_basin'):
    """Evaluate LightGBM by dynamically finding candidates"""
    
    # Load the trained model
    try:
        with open('models/lightgbm_ranker.pkl', 'rb') as f:
            lgb_dict = pickle.load(f)
        # Extract the actual model from the dictionary
        lgb_model = lgb_dict['model'] if isinstance(lgb_dict, dict) else lgb_dict
        logger.info("Loaded trained LightGBM model")
    except FileNotFoundError:
        logger.error("LightGBM model not found at models/lightgbm_ranker.pkl")
        return [], []
    
    db = DatabaseConnector()
    errors = []
    predictions = []
    successful = 0
    failed = 0
    no_analogs = 0
    
    logger.info(f"Evaluating LightGBM model for {basin}...")
    
    for idx, well in test_wells.iterrows():
        try:
            # Dynamically find candidates from historical wells in early_rates
            query = """
            WITH target AS (
                SELECT 
                    %s::text as target_well_id,
                    %s::text as formation,
                    %s::float as lateral_length,
                    %s::float as proppant_per_ft,
                    %s::float as fluid_per_ft,
                    %s::float as lat,
                    %s::float as lon,
                    %s::date as first_prod_date
            )
            SELECT 
                c.well_id as candidate_well_id,
                c.oil_m1_9,
                c.lateral_length,
                c.proppant_per_ft,
                c.fluid_per_ft,
                ST_Distance(
                    c.geom::geography, 
                    ST_SetSRID(ST_MakePoint(t.lon, t.lat), 4326)::geography
                ) / 1609.34 as distance_mi,
                c.lateral_length / t.lateral_length as length_ratio,
                1 as formation_match,  -- Always 1 now due to WHERE filter
                CASE WHEN c.operator_name = %s THEN 1 ELSE 0 END as same_operator,
                EXTRACT(YEAR FROM t.first_prod_date) - EXTRACT(YEAR FROM c.first_prod_date) as vintage_gap_years,
                c.proppant_per_ft / NULLIF(t.proppant_per_ft, 0) as ppf_ratio,
                c.fluid_per_ft / NULLIF(t.fluid_per_ft, 0) as fpf_ratio
            FROM data.early_rates c, target t
            WHERE c.basin_name = %s
            AND c.first_prod_date < t.first_prod_date  -- Only historical wells
            AND c.formation = t.formation  -- STRICT: Only same formation
            AND ST_DWithin(
                c.geom::geography, 
                ST_SetSRID(ST_MakePoint(t.lon, t.lat), 4326)::geography, 
                20 * 1609.34  -- 20 miles
            )
            ORDER BY distance_mi
            LIMIT 500
            """
            
            with db.get_connection() as conn:
                candidates = pd.read_sql_query(query, conn, params=(
                    well['well_id'], well['formation'], well['lateral_length'],
                    well['proppant_per_ft'], well['fluid_per_ft'],
                    well['lat'], well['lon'], well['first_prod_date'],
                    well['operator_name'], basin
                ))
            
            if len(candidates) == 0:
                no_analogs += 1
                failed += 1
                logger.debug(f"No candidates found for well {well['well_id']}")
                continue
            
            # Build features using centralized function
            target_well_dict = {
                'lateral_length': well['lateral_length'],
                'proppant_per_ft': well['proppant_per_ft'],
                'formation': well['formation'],
                'operator_name': well['operator_name'],
                'first_prod_date': well['first_prod_date']
            }
            
            # Rename columns to match what ranking_features expects
            candidates_renamed = candidates.copy()
            candidates_renamed['lateral_length'] = candidates['lateral_length']
            candidates_renamed['proppant_per_ft'] = candidates['proppant_per_ft']
            
            features = build_ranking_features(candidates_renamed, target_well_dict)
            
            # Get expected feature names from the model
            if isinstance(lgb_dict, dict) and 'feature_columns' in lgb_dict:
                feature_cols = lgb_dict['feature_columns']
            else:
                feature_cols = get_feature_names()
            
            # Ensure features are in the right order
            features = features[feature_cols]
            
            # Score with LightGBM
            scores = lgb_model.predict(features)
            candidates['lgb_score'] = scores
            
            # Select top 20 by score
            top_candidates = candidates.nlargest(20, 'lgb_score')
            
            # Apply warping and average production from top candidates
            # Get warping coefficients (matching training)
            kappa_length = 0.6  # From basin_config
            kappa_proppant = 0.2  # From basin_config
            
            oil_curves = []
            for _, cand in top_candidates.iterrows():
                # Convert to float to handle Decimal types from PostgreSQL
                oil_curve = np.array(cand['oil_m1_9'], dtype=float)
                
                # Apply warping (matching training logic from lightgbm_ranker.py)
                # Length warping
                length_ratio = well['lateral_length'] / cand['lateral_length'] if cand['lateral_length'] > 0 else 1.0
                length_scaler = np.power(length_ratio, kappa_length)
                length_scaler = min(1.3, max(0.7, length_scaler))  # Clip to [0.7, 1.3]
                
                # Proppant warping
                if well['proppant_per_ft'] > 0 and cand['proppant_per_ft'] > 0:
                    ppf_ratio = well['proppant_per_ft'] / cand['proppant_per_ft']
                    ppf_scaler = np.power(ppf_ratio, kappa_proppant)
                    ppf_scaler = min(1.3, max(0.7, ppf_scaler))  # Clip to [0.7, 1.3]
                else:
                    ppf_scaler = 1.0
                
                # Apply combined warping
                warped_curve = oil_curve * length_scaler * ppf_scaler
                oil_curves.append(warped_curve)
            
            pred_oil = np.mean(oil_curves, axis=0)
            
            # Calculate error (convert to float to handle Decimal types from PostgreSQL)
            actual = np.array(well['oil_m1_9'], dtype=float)
            error = calculate_weighted_mae(actual, pred_oil)
            
            errors.append(error)
            predictions.append({
                'well_id': well['well_id'],
                'formation': well['formation'],
                'actual': actual,
                'predicted': pred_oil,
                'error': error,
                'n_analogs': len(top_candidates)
            })
            successful += 1
            
            if successful % 10 == 0:
                logger.info(f"Processed {successful} wells, median error: {np.median(errors):.0f} bbls")
                
        except Exception as e:
            logger.error(f"Error processing well {well['well_id']}: {e}")
            failed += 1
            continue
    
    logger.info(f"LightGBM complete: {successful} successful, {failed} failed ({no_analogs} had no analogs)")
    return errors, predictions

def compare_results(baseline_errors, lightgbm_errors, basin, year):
    """Generate comparison statistics and visualization"""
    
    if not baseline_errors or not lightgbm_errors:
        logger.error("Not enough data to compare models")
        return None, None
    
    print("\n" + "="*60)
    print(f"MODEL COMPARISON - {basin.upper()} - H1 {year}")
    print("="*60)
    
    # Calculate statistics
    baseline_stats = {
        'median': np.median(baseline_errors),
        'mean': np.mean(baseline_errors),
        'std': np.std(baseline_errors),
        'p10': np.percentile(baseline_errors, 10),
        'p90': np.percentile(baseline_errors, 90),
        'n_wells': len(baseline_errors)
    }
    
    lightgbm_stats = {
        'median': np.median(lightgbm_errors),
        'mean': np.mean(lightgbm_errors),
        'std': np.std(lightgbm_errors),
        'p10': np.percentile(lightgbm_errors, 10),
        'p90': np.percentile(lightgbm_errors, 90),
        'n_wells': len(lightgbm_errors)
    }
    
    # Print comparison table
    print(f"\nWeighted MAE (bbls/month):")
    print(f"{'Metric':<15} {'Baseline':>12} {'LightGBM':>12} {'Difference':>15}")
    print("-" * 55)
    
    for metric in ['median', 'mean', 'p10', 'p90']:
        baseline_val = baseline_stats[metric]
        lightgbm_val = lightgbm_stats[metric]
        diff = lightgbm_val - baseline_val
        diff_pct = (diff / baseline_val) * 100 if baseline_val > 0 else 0
        
        print(f"{metric.upper():<15} {baseline_val:>12.0f} {lightgbm_val:>12.0f} "
              f"{diff:>+8.0f} ({diff_pct:+.1f}%)")
    
    print(f"\nWells evaluated: {baseline_stats['n_wells']}")
    
    # Determine winner
    print("\n" + "="*60)
    if baseline_stats['median'] < lightgbm_stats['median']:
        improvement = ((lightgbm_stats['median'] - baseline_stats['median']) / 
                      lightgbm_stats['median'] * 100)
        print(f"✓ WINNER: BASELINE (lower error by {improvement:.1f}%)")
    else:
        improvement = ((baseline_stats['median'] - lightgbm_stats['median']) / 
                      baseline_stats['median'] * 100)
        print(f"✓ WINNER: LIGHTGBM (lower error by {improvement:.1f}%)")
    print("="*60)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot comparison
    box_data = [baseline_errors, lightgbm_errors]
    bp = ax1.boxplot(box_data, labels=['Baseline', 'LightGBM'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Weighted MAE (bbls/month)')
    ax1.set_title(f'{basin.upper()} - H1 {year}')
    ax1.grid(True, alpha=0.3)
    
    # Histogram comparison
    ax2.hist(baseline_errors, bins=30, alpha=0.5, label='Baseline', color='blue', density=True)
    ax2.hist(lightgbm_errors, bins=30, alpha=0.5, label='LightGBM', color='green', density=True)
    ax2.set_xlabel('Weighted MAE (bbls/month)')
    ax2.set_ylabel('Density')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Model Comparison Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f'comparison_{basin}_{year}.png'
    plt.savefig(filename, dpi=100)
    print(f"\nVisualization saved to {filename}")
    plt.close()
    
    return baseline_stats, lightgbm_stats

def main():
    """Main execution function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Compare Baseline vs LightGBM models')
    parser.add_argument('--basin', type=str, default='both',
                        help='Basin to evaluate (dj_basin, bakken, or both)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Test year (default: 2024 - out of sample)')
    args = parser.parse_args()
    
    # Initialize database
    db = DatabaseConnector()
    
    basins = ['dj_basin', 'bakken'] if args.basin == 'both' else [args.basin]
    
    all_results = {}
    
    for basin in basins:
        print(f"\n{'='*60}")
        print(f"Comparing models for {basin}")
        print(f"Test set: ALL H1 {args.year} wells")
        print('='*60)
        
        # Load test wells from model_wells/model_prod, NOT early_rates
        test_wells = load_test_wells(db, basin=basin, year=args.year)
        
        if len(test_wells) == 0:
            logger.error(f"No test wells found for {basin} H1 {args.year}")
            continue
        
        print(f"\nEvaluating {len(test_wells)} test wells...")
        
        # Evaluate both models
        baseline_errors, baseline_preds = evaluate_baseline(test_wells, basin=basin)
        lightgbm_errors, lightgbm_preds = evaluate_lightgbm_simple(test_wells, basin=basin)
        
        # Compare results
        if baseline_errors and lightgbm_errors:
            baseline_stats, lightgbm_stats = compare_results(
                baseline_errors, lightgbm_errors, basin, args.year
            )
            
            all_results[basin] = {
                'baseline': baseline_stats,
                'lightgbm': lightgbm_stats,
                'n_wells': len(test_wells)
            }
    
    # Summary across basins
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("SUMMARY ACROSS BASINS")
        print("="*60)
        
        for basin, results in all_results.items():
            print(f"\n{basin.upper()}:")
            print(f"  Baseline median MAE: {results['baseline']['median']:.0f} bbls")
            print(f"  LightGBM median MAE: {results['lightgbm']['median']:.0f} bbls")
            
            if results['baseline']['median'] < results['lightgbm']['median']:
                print(f"  Winner: BASELINE")
            else:
                print(f"  Winner: LIGHTGBM")

if __name__ == "__main__":
    main()