"""
Step 7: Production Inference Pipeline
Provides a clean interface for making analog-based production forecasts for new wells
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.db_connector import DatabaseConnector
from src.models.baseline import BaselinePicker

# Optional import for LightGBM (may not be implemented yet)
try:
    from src.models.lightgbm_ranker import LightGBMAnalogRanker
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LightGBMAnalogRanker = None
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnalogPredictor:
    """Production-ready analog predictor for new wells"""
    
    def __init__(self, model_type: str = 'lightgbm', model_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_type: 'lightgbm' or 'baseline'
            model_path: Optional path to saved model
        """
        self.db = DatabaseConnector()
        self.model_type = model_type
        
        # Load appropriate model
        if model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM ranker not available. Use model_type='baseline' instead.")
            self.model = LightGBMAnalogRanker()
            if model_path:
                self.model.load(model_path)
            else:
                self.model.load('models/lightgbm_ranker.pkl')
        elif model_type == 'baseline':
            self.model = BaselinePicker()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized {model_type} predictor")
    
    def predict_new_well(self, 
                        lat: float,
                        lon: float, 
                        formation: str,
                        lateral_length: float,
                        proppant_per_ft: float,
                        fluid_per_ft: float,
                        operator: Optional[str] = None,
                        planned_date: Optional[str] = None) -> Dict:
        """
        Make production forecast for a new well
        
        Args:
            lat: Surface latitude
            lon: Surface longitude
            formation: Formation name
            lateral_length: Completed lateral length (ft)
            proppant_per_ft: Proppant loading (lb/ft)
            fluid_per_ft: Fluid loading (gal/ft)
            operator: Optional operator name
            planned_date: Optional planned completion date (YYYY-MM-DD)
            
        Returns:
            Dictionary with forecast and analog information
        """
        logger.info(f"Predicting for new well at ({lat:.4f}, {lon:.4f})")
        
        # Step 2: Score and rank candidates
        if self.model_type == 'lightgbm':
            # Step 1: Find analog candidates for LightGBM
            candidates = self._find_candidates(
                lat, lon, formation, lateral_length, 
                proppant_per_ft, fluid_per_ft, planned_date
            )
            
            if len(candidates) == 0:
                return {
                    'error': 'No suitable analog candidates found',
                    'lat': lat,
                    'lon': lon,
                    'formation': formation
                }
            
            logger.info(f"Found {len(candidates)} analog candidates")
            
            result = self._predict_lightgbm(
                candidates, lateral_length, proppant_per_ft, 
                fluid_per_ft, operator
            )
        else:
            # Use BaselinePicker's integrated pipeline (does its own candidate finding)
            result = self.model.predict(
                target_lat=lat,
                target_lon=lon,
                target_formation=formation,
                target_lateral_length=lateral_length,
                target_proppant_per_ft=proppant_per_ft,
                cutoff_date=planned_date
            )
            
            # Transform to expected format if successful
            if 'error' not in result:
                result = self._format_baseline_result(result)
        
        # Add input parameters
        result['input'] = {
            'lat': lat,
            'lon': lon,
            'formation': formation,
            'lateral_length': lateral_length,
            'proppant_per_ft': proppant_per_ft,
            'fluid_per_ft': fluid_per_ft,
            'operator': operator,
            'planned_date': planned_date
        }
        
        return result
    
    def _find_candidates(self, lat: float, lon: float, formation: str,
                        lateral_length: float, proppant_per_ft: float,
                        fluid_per_ft: float, planned_date: Optional[str]) -> pd.DataFrame:
        """Find analog candidates for new well"""
        
        # Date filter - only use wells before planned date
        date_filter = ""
        if planned_date:
            date_filter = f"AND er.first_prod_date <= '{planned_date}'"
        
        query = f"""
        WITH target_location AS (
            SELECT ST_SetSRID(ST_MakePoint(%s, %s), 4326) as geom
        )
        SELECT
            er.well_id,
            er.first_prod_date,
            er.formation,
            er.operator,
            er.lateral_length,
            er.proppant_per_ft,
            er.fluid_per_ft,
            er.oil_m1_9,
            er.gas_m1_9,
            ST_Distance(er.geom::geography, tl.geom::geography) / 1609.34 as distance_mi,
            %s / er.lateral_length as length_ratio,
            %s - er.lateral_length as delta_length,
            CASE WHEN UPPER(er.formation) = UPPER(%s) THEN 1 ELSE 0 END as formation_match,
            %s / NULLIF(er.proppant_per_ft, 0) as ppf_ratio,
            %s / NULLIF(er.fluid_per_ft, 0) as fpf_ratio,
            EXTRACT(YEAR FROM NOW()) - EXTRACT(YEAR FROM er.first_prod_date) as vintage_years
        FROM data.early_rates er, target_location tl
        WHERE er.oil_m1_9 IS NOT NULL
        AND er.lateral_length BETWEEN %s * 0.8 AND %s * 1.2
        AND ST_DWithin(er.geom::geography, tl.geom::geography, 15 * 1609.34)  -- 15 miles
        {date_filter}
        ORDER BY distance_mi
        LIMIT 500
        """
        
        with self.db.get_connection() as conn:
            candidates = pd.read_sql(
                query, conn,
                params=(lon, lat, lateral_length, lateral_length, 
                       formation, proppant_per_ft, fluid_per_ft,
                       lateral_length, lateral_length)
            )
        
        return candidates
    
    def _format_baseline_result(self, baseline_result: Dict) -> Dict:
        """Format BaselinePicker result to match AnalogPredictor expected format"""
        
        return {
            'model': 'baseline',
            'forecast': {
                'p50_oil_monthly': baseline_result['p50_oil_monthly'],
                'p10_oil_monthly': baseline_result['p10_oil_monthly'],
                'p90_oil_monthly': baseline_result['p90_oil_monthly'],
                'p50_cum_9mo': baseline_result['p50_cum_9mo'],
                'p10_cum_9mo': baseline_result['p10_cum_9mo'],
                'p90_cum_9mo': baseline_result['p90_cum_9mo']
            },
            'n_analogs': baseline_result['n_analogs'],
            'analogs': baseline_result['analogs'],
            'metadata': baseline_result['metadata']
        }
    
    def _predict_lightgbm(self, candidates: pd.DataFrame, 
                         lateral_length: float, proppant_per_ft: float,
                         fluid_per_ft: float, operator: Optional[str]) -> Dict:
        """Make prediction using LightGBM model"""
        
        # Engineer features
        features = pd.DataFrame()
        
        # Distance features
        features['distance_miles'] = candidates['distance_mi']
        features['dist_0_5'] = (candidates['distance_mi'] <= 5).astype(int)
        features['dist_5_10'] = ((candidates['distance_mi'] > 5) & 
                                 (candidates['distance_mi'] <= 10)).astype(int)
        features['dist_10_15'] = ((candidates['distance_mi'] > 10) & 
                                  (candidates['distance_mi'] <= 15)).astype(int)
        
        # Lateral length features
        features['length_ratio'] = candidates['length_ratio']
        features['log_length_ratio'] = np.log1p(candidates['length_ratio'])
        features['delta_length'] = candidates['delta_length']
        features['target_lateral'] = lateral_length
        features['candidate_lateral'] = candidates['lateral_length']
        
        # Design parameters
        features['ppf_ratio'] = candidates['ppf_ratio'].fillna(1.0)
        features['fpf_ratio'] = candidates['fpf_ratio'].fillna(1.0)
        features['log_ppf_ratio'] = np.log1p(features['ppf_ratio'])
        features['log_fpf_ratio'] = np.log1p(features['fpf_ratio'])
        
        # Categorical matches
        features['formation_match'] = candidates['formation_match']
        features['same_operator'] = 0  # Unknown operator
        if operator:
            features['same_operator'] = (candidates['operator'] == operator).astype(int)
        
        # Vintage features
        features['vintage_gap'] = candidates['vintage_years']
        features['vintage_gap_squared'] = features['vintage_gap'] ** 2
        features['recent_vintage'] = (features['vintage_gap'] <= 2).astype(int)
        features['target_year'] = datetime.now().year
        features['candidate_year'] = features['target_year'] - features['vintage_gap']
        
        # Interaction features
        features['formation_distance'] = features['formation_match'] * features['distance_miles']
        features['operator_distance'] = features['same_operator'] * features['distance_miles']
        
        # Score candidates
        scores = self.model.model.predict(
            features[self.model.feature_names], 
            num_iteration=self.model.model.best_iteration
        )
        
        candidates['score'] = scores
        candidates = candidates.sort_values('score', ascending=False)
        
        # Select top-k with operator diversity
        selected = []
        operator_counts = {}
        
        for _, row in candidates.iterrows():
            op = row['operator']
            if operator_counts.get(op, 0) < self.model.max_per_operator:
                selected.append(row)
                operator_counts[op] = operator_counts.get(op, 0) + 1
                if len(selected) >= self.model.k_analogs:
                    break
        
        selected_df = pd.DataFrame(selected)
        
        # Warp and average production curves
        warped_curves = []
        for _, row in selected_df.iterrows():
            oil_curve = np.array([float(v) for v in row['oil_m1_9']])
            
            # Apply warping
            length_scaler = min(1.3, max(0.7, lateral_length / row['lateral_length']))
            ppf_scaler = min(1.3, max(0.7, np.power(
                proppant_per_ft / (row['proppant_per_ft'] + 1), 0.2
            )))
            
            warped = oil_curve * length_scaler * ppf_scaler
            warped_curves.append(warped)
        
        # Weighted average using softmax of scores
        scores_selected = selected_df['score'].values
        weights = np.exp(scores_selected / 0.4)
        weights = weights / weights.sum()
        
        p50_curve = np.average(warped_curves, weights=weights, axis=0)
        
        # P10/P90 via percentiles
        p10_curve = np.percentile(warped_curves, 10, axis=0)
        p90_curve = np.percentile(warped_curves, 90, axis=0)
        
        # Prepare analog list
        analogs = []
        for _, row in selected_df.head(10).iterrows():
            analogs.append({
                'well_id': row['well_id'],
                'distance_mi': round(row['distance_mi'], 1),
                'score': round(row['score'], 2),
                'operator': row['operator'],
                'formation': row['formation'],
                'lateral_length': row['lateral_length'],
                'vintage_year': int(row['first_prod_date'].year) if pd.notna(row['first_prod_date']) else None,
                'weight': round(weights[len(analogs)] * 100, 1) if len(analogs) < len(weights) else 0
            })
        
        return {
            'model': 'lightgbm',
            'forecast': {
                'p50_oil_monthly': p50_curve.tolist(),
                'p10_oil_monthly': p10_curve.tolist(),
                'p90_oil_monthly': p90_curve.tolist(),
                'p50_cum_9mo': float(np.sum(p50_curve)),
                'p10_cum_9mo': float(np.sum(p10_curve)),
                'p90_cum_9mo': float(np.sum(p90_curve))
            },
            'n_analogs': len(selected_df),
            'analogs': analogs,
            'metadata': {
                'avg_distance_mi': round(selected_df['distance_mi'].mean(), 1),
                'formation_match_pct': round(selected_df['formation_match'].mean() * 100, 1),
                'avg_vintage_gap': round(selected_df['vintage_years'].mean(), 1)
            }
        }
    
    def predict_batch(self, wells: List[Dict], output_file: Optional[str] = None) -> List[Dict]:
        """
        Make predictions for multiple wells
        
        Args:
            wells: List of well dictionaries with required fields
            output_file: Optional file to save results
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, well in enumerate(wells, 1):
            logger.info(f"Processing well {i}/{len(wells)}")
            
            try:
                result = self.predict_new_well(
                    lat=well['lat'],
                    lon=well['lon'],
                    formation=well['formation'],
                    lateral_length=well['lateral_length'],
                    proppant_per_ft=well.get('proppant_per_ft', 2000),
                    fluid_per_ft=well.get('fluid_per_ft', 50),
                    operator=well.get('operator'),
                    planned_date=well.get('planned_date')
                )
                
                # Add well ID if provided
                if 'well_id' in well:
                    result['well_id'] = well['well_id']
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing well {i}: {e}")
                results.append({
                    'error': str(e),
                    'well_id': well.get('well_id', f'well_{i}')
                })
        
        # Save if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def export_forecast_csv(self, result: Dict, output_file: str) -> None:
        """
        Export forecast to CSV format
        
        Args:
            result: Prediction result dictionary
            output_file: Output CSV file path
        """
        if 'error' in result:
            logger.error(f"Cannot export - error in result: {result['error']}")
            return
        
        # Create DataFrame with monthly forecasts
        df = pd.DataFrame({
            'month': range(1, 10),
            'p10_oil': result['forecast']['p10_oil_monthly'],
            'p50_oil': result['forecast']['p50_oil_monthly'],
            'p90_oil': result['forecast']['p90_oil_monthly']
        })
        
        # Add metadata
        df['lat'] = result['input']['lat']
        df['lon'] = result['input']['lon']
        df['formation'] = result['input']['formation']
        df['lateral_length'] = result['input']['lateral_length']
        df['n_analogs'] = result['n_analogs']
        
        df.to_csv(output_file, index=False)
        logger.info(f"Forecast exported to {output_file}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analog-based production predictor')
    parser.add_argument('--model', choices=['lightgbm', 'baseline'], default='lightgbm',
                       help='Model type to use')
    parser.add_argument('--lat', type=float, help='Surface latitude')
    parser.add_argument('--lon', type=float, help='Surface longitude')
    parser.add_argument('--formation', type=str, help='Formation name')
    parser.add_argument('--lateral', type=float, help='Lateral length (ft)')
    parser.add_argument('--ppf', type=float, default=2000, help='Proppant per ft')
    parser.add_argument('--fpf', type=float, default=50, help='Fluid per ft')
    parser.add_argument('--operator', type=str, help='Operator name')
    parser.add_argument('--date', type=str, help='Planned date (YYYY-MM-DD)')
    parser.add_argument('--batch', type=str, help='Batch input JSON file')
    parser.add_argument('--output', type=str, help='Output file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AnalogPredictor(model_type=args.model)
    
    if args.batch:
        # Batch prediction
        with open(args.batch, 'r') as f:
            wells = json.load(f)
        results = predictor.predict_batch(wells, args.output)
        
        # Summary
        success = [r for r in results if 'error' not in r]
        print(f"\nProcessed {len(wells)} wells")
        print(f"Success: {len(success)}")
        print(f"Failures: {len(results) - len(success)}")
        
        if success:
            avg_p50 = np.mean([r['forecast']['p50_cum_9mo'] for r in success])
            print(f"Average P50 9-month cum: {avg_p50:,.0f} bbls")
    
    elif args.lat and args.lon and args.formation and args.lateral:
        # Single prediction
        result = predictor.predict_new_well(
            lat=args.lat,
            lon=args.lon,
            formation=args.formation,
            lateral_length=args.lateral,
            proppant_per_ft=args.ppf,
            fluid_per_ft=args.fpf,
            operator=args.operator,
            planned_date=args.date
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n{'='*60}")
            print(f"PRODUCTION FORECAST - {args.model.upper()} MODEL")
            print(f"{'='*60}")
            print(f"Location: ({args.lat:.4f}, {args.lon:.4f})")
            print(f"Formation: {args.formation}")
            print(f"Lateral: {args.lateral:,.0f} ft")
            print(f"\n9-Month Cumulative Oil (bbls):")
            print(f"  P10: {result['forecast']['p10_cum_9mo']:,.0f}")
            print(f"  P50: {result['forecast']['p50_cum_9mo']:,.0f}")
            print(f"  P90: {result['forecast']['p90_cum_9mo']:,.0f}")
            print(f"\nUsed {result['n_analogs']} analogs")
            print(f"Average distance: {result['metadata']['avg_distance_mi']:.1f} miles")
            print(f"Formation match: {result['metadata']['formation_match_pct']:.0f}%")
            
            if result['analogs']:
                print(f"\nTop 5 Analogs:")
                for analog in result['analogs'][:5]:
                    print(f"  - {analog['well_id']}: "
                          f"{analog['distance_mi']} mi, "
                          f"{analog['operator']}, "
                          f"{analog.get('weight', 0):.1f}% weight")
            
            # Save if requested
            if args.output:
                if args.output.endswith('.csv'):
                    predictor.export_forecast_csv(result, args.output)
                else:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"\nFull results saved to {args.output}")
    
    else:
        print("Provide either --batch or individual well parameters")
        parser.print_help()


if __name__ == "__main__":
    main()