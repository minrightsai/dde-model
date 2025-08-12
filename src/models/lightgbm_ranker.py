"""
Step 6: LightGBM LambdaRank Model
Learning-to-rank model for improved analog selection
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import logging
import joblib
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.db_connector import DatabaseConnector
from src.features.feature_builder import FeatureBuilder
from src.features.ranking_features import build_ranking_features, get_feature_names
from src.models.baseline import BaselinePicker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LightGBMAnalogRanker:
    """
    LightGBM LambdaRank model for learning analog selection patterns.
    Improves upon baseline by learning from historical analog performance.
    """
    
    def __init__(self,
                 basin: Optional[str] = None,
                 lgb_params: Optional[Dict] = None,
                 top_k: int = 20,
                 max_per_operator: int = 3,
                 kappa_length: float = 0.6,
                 kappa_proppant: float = 0.2):
        """
        Initialize LightGBM ranker
        
        Args:
            basin: Basin to train/predict for ('dj_basin' or 'bakken')
            lgb_params: LightGBM parameters (uses defaults if None)
            top_k: Number of top analogs to use
            max_per_operator: Maximum analogs per operator
            kappa_length: Warping exponent for lateral length
            kappa_proppant: Warping exponent for proppant
        """
        self.basin = basin
        self.db = DatabaseConnector()
        self.feature_builder = FeatureBuilder()
        self.baseline_picker = BaselinePicker()  # For candidate generation
        
        # Model parameters
        self.top_k = top_k
        self.max_per_operator = max_per_operator
        self.kappa_length = kappa_length
        self.kappa_proppant = kappa_proppant
        
        # LightGBM parameters
        if lgb_params is None:
            self.lgb_params = {
                'objective': 'lambdarank',
                'metric': ['ndcg', 'map'],
                'lambdarank_truncation_level': 20,
                'label_gain': [0, 1, 3, 7],  # Graded relevance gains
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'max_depth': 7,
                'num_threads': -1,
                'verbosity': -1,
                'random_state': 42
            }
        else:
            self.lgb_params = lgb_params
        
        self.model = None
        self.feature_columns = None
        
    def prepare_training_data(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare training data with features and graded labels
        
        Args:
            limit: Optional limit on number of target wells
            
        Returns:
            Tuple of (features_df, labels, groups)
        """
        logger.info("Loading training data...")
        
        # Load candidate pairs with features
        basin_filter = f"AND er.basin_name = '{self.basin}'" if self.basin else ""
        basin_filter_ac = f"AND ac.basin_name = '{self.basin}'" if self.basin else ""
        
        query = """
        WITH training_targets AS (
            SELECT DISTINCT ac.target_well_id
            FROM data.analog_candidates ac
            JOIN data.early_rates er ON ac.target_well_id = er.well_id
            WHERE EXTRACT(YEAR FROM er.first_prod_date) BETWEEN 2016 AND 2023
            {basin_filter}
            ORDER BY ac.target_well_id
            {limit_clause}
        ),
        candidate_pairs AS (
            SELECT 
                ac.target_well_id,
                ac.candidate_well_id,
                ac.distance_mi,
                ac.length_ratio,
                ac.delta_length,
                ac.formation_match,
                ac.same_operator,
                ac.vintage_gap_years,
                ac.ppf_ratio,
                ac.fpf_ratio,
                t.oil_m1_9 as target_oil,
                t.lateral_length as target_lateral,
                t.proppant_per_ft as target_ppf,
                ac.target_formation,
                ac.target_operator,
                c.oil_m1_9 as candidate_oil,
                c.lateral_length as candidate_lateral,
                c.proppant_per_ft as candidate_ppf,
                ac.candidate_formation,
                ac.candidate_operator,
                ce.oil_embedding as candidate_embedding
            FROM data.analog_candidates ac
            JOIN training_targets tt ON ac.target_well_id = tt.target_well_id
            JOIN data.early_rates t ON ac.target_well_id = t.well_id
            JOIN data.early_rates c ON ac.candidate_well_id = c.well_id
            LEFT JOIN data.curve_embeddings ce ON c.well_id = ce.well_id
            WHERE ac.formation_match = true  -- ONLY same-formation pairs for training
            {basin_filter_ac}
        )
        SELECT * FROM candidate_pairs
        ORDER BY target_well_id, candidate_well_id
        """.format(
            basin_filter=basin_filter,
            basin_filter_ac=basin_filter_ac,
            limit_clause=f"LIMIT {limit}" if limit else ""
        )
        
        with self.db.get_connection() as conn:
            data = pd.read_sql(query, conn)
        
        # Convert decimal columns to float (handle PostgreSQL Decimal types)
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col], errors='ignore')
                except:
                    pass
        
        logger.info(f"Loaded {len(data)} candidate pairs from {data['target_well_id'].nunique()} target wells")
        
        # Build features using centralized function
        # Create target well dicts for each row
        all_features = []
        for target_id in data['target_well_id'].unique():
            target_data = data[data['target_well_id'] == target_id].iloc[0]
            target_well = {
                'lateral_length': target_data['target_lateral'],
                'proppant_per_ft': target_data['target_ppf'],
                'formation': target_data['target_formation'],
                'operator_name': target_data['target_operator'],
                'first_prod_date': None  # Not needed since vintage_gap is pre-computed
            }
            candidates = data[data['target_well_id'] == target_id].copy()
            features_batch = build_ranking_features(candidates, target_well)
            all_features.append(features_batch)
        
        features = pd.concat(all_features, ignore_index=True)
        self.feature_columns = get_feature_names()
        
        # Calculate labels based on revenue-weighted error
        labels = self._calculate_graded_labels(data)
        
        # Create groups for ranking (one group per target well)
        groups = data.groupby('target_well_id').size().values
        
        return features, labels, groups
    
    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build features for candidate pairs
        
        Args:
            data: Raw candidate pair data
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame()
        
        # Distance features
        features['distance_mi'] = data['distance_mi']
        features['distance_km'] = data['distance_mi'] * 1.60934
        features['log_distance'] = np.log1p(data['distance_mi'])
        
        # Lateral length features
        features['length_ratio'] = data['length_ratio']
        features['length_delta'] = data['delta_length']
        features['log_length_ratio'] = np.log(data['length_ratio'].clip(lower=0.1))
        features['abs_log_length_ratio'] = np.abs(features['log_length_ratio'])
        
        # Design parameter ratios
        ppf_ratio = data['target_ppf'] / (data['candidate_ppf'] + 1)
        features['ppf_ratio'] = ppf_ratio
        features['log_ppf_ratio'] = np.log(ppf_ratio.clip(lower=0.1))
        
        # Formation and operator matching (already in data)
        features['same_formation'] = data['formation_match'].astype(int)
        features['same_operator'] = data['same_operator'].astype(int)
        
        # Vintage gap
        features['vintage_gap_years'] = data['vintage_gap_years'].fillna(0)
        features['abs_vintage_gap'] = np.abs(features['vintage_gap_years'])
        
        # Embedding features if available
        if 'candidate_embedding' in data.columns and data['candidate_embedding'].notna().any():
            # Extract PCA components
            embeddings = np.vstack([
                np.array(emb) if emb is not None else np.zeros(4)
                for emb in data['candidate_embedding']
            ])
            for i in range(embeddings.shape[1]):
                features[f'embedding_pc{i+1}'] = embeddings[:, i]
        
        # Interaction features
        features['distance_x_formation'] = features['distance_mi'] * features['same_formation']
        features['distance_x_operator'] = features['distance_mi'] * features['same_operator']
        features['length_x_ppf'] = features['abs_log_length_ratio'] * features['log_ppf_ratio']
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def _calculate_graded_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate graded relevance labels based on revenue-weighted error
        
        Args:
            data: Candidate pair data with production arrays
            
        Returns:
            Array of graded labels (0-3)
        """
        labels = []
        
        # Revenue weights
        oil_price = 70
        discount_rate = 0.10
        weights = np.array([oil_price / (1 + discount_rate/12)**i for i in range(9)])
        
        for target_id in data['target_well_id'].unique():
            target_data = data[data['target_well_id'] == target_id].copy()
            
            # Get target production
            target_oil = np.array([float(v) for v in target_data.iloc[0]['target_oil']])
            
            errors = []
            for _, row in target_data.iterrows():
                # Get candidate production
                candidate_oil = np.array([float(v) for v in row['candidate_oil']])
                
                # Apply warping (handle decimal types)
                length_ratio = float(row['length_ratio'])
                length_scaler = np.power(length_ratio, self.kappa_length)
                length_scaler = min(1.3, max(0.7, length_scaler))
                
                ppf_ratio = float(row['target_ppf']) / (float(row['candidate_ppf']) + 1)
                ppf_scaler = np.power(ppf_ratio, self.kappa_proppant)
                ppf_scaler = min(1.3, max(0.7, ppf_scaler))
                
                warped = candidate_oil * length_scaler * ppf_scaler
                
                # Calculate revenue-weighted error
                abs_errors = np.abs(target_oil - warped)
                rev_weighted_mae = np.sum(weights * abs_errors) / np.sum(weights)
                errors.append(rev_weighted_mae)
            
            # Convert errors to graded labels (0-3)
            errors = np.array(errors)
            percentiles = np.percentile(errors, [10, 30, 60])
            
            group_labels = np.zeros(len(errors))
            group_labels[errors <= percentiles[0]] = 3  # Top 10% - excellent
            group_labels[(errors > percentiles[0]) & (errors <= percentiles[1])] = 2  # Next 20% - good
            group_labels[(errors > percentiles[1]) & (errors <= percentiles[2])] = 1  # Next 30% - fair
            # Rest remain 0 (poor)
            
            labels.extend(group_labels)
        
        return np.array(labels)
    
    def train(self, features: pd.DataFrame, labels: np.ndarray, groups: np.ndarray,
              valid_features: Optional[pd.DataFrame] = None,
              valid_labels: Optional[np.ndarray] = None,
              valid_groups: Optional[np.ndarray] = None,
              num_boost_round: int = 200,
              early_stopping_rounds: int = 20):
        """
        Train the LightGBM ranker
        
        Args:
            features: Training features
            labels: Graded labels
            groups: Group sizes for ranking
            valid_features: Validation features
            valid_labels: Validation labels
            valid_groups: Validation groups
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
        """
        logger.info(f"Training LightGBM ranker with {len(features)} samples...")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(features, label=labels, group=groups)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if valid_features is not None:
            valid_data = lgb.Dataset(valid_features, label=valid_labels, group=valid_groups, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Train model
        callbacks = [lgb.log_evaluation(10)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Log feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    def score_candidates(self, candidates: pd.DataFrame, target_well: Dict) -> np.ndarray:
        """
        Score candidates using the trained model
        
        Args:
            candidates: DataFrame of candidate analogs
            target_well: Target well properties
            
        Returns:
            Array of scores
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Build features for candidates
        features = pd.DataFrame()
        
        # Basic features
        features['distance_mi'] = candidates['distance_mi']
        features['distance_km'] = candidates['distance_mi'] * 1.60934
        features['log_distance'] = np.log1p(candidates['distance_mi'])
        
        features['length_ratio'] = candidates['length_ratio']
        features['length_delta'] = candidates.get('delta_length', 
                                                  candidates['lateral_length'] - target_well['lateral_length'])
        features['log_length_ratio'] = np.log(candidates['length_ratio'].clip(lower=0.1))
        features['abs_log_length_ratio'] = np.abs(features['log_length_ratio'])
        
        # Design parameters
        if 'proppant_per_ft' in candidates.columns and 'proppant_per_ft' in target_well:
            ppf_ratio = target_well['proppant_per_ft'] / (candidates['proppant_per_ft'] + 1)
            features['ppf_ratio'] = ppf_ratio
            features['log_ppf_ratio'] = np.log(ppf_ratio.clip(lower=0.1))
        else:
            features['ppf_ratio'] = 1.0
            features['log_ppf_ratio'] = 0.0
        
        # Matching features
        features['same_formation'] = (candidates['formation'] == target_well['formation']).astype(int)
        features['same_operator'] = (candidates.get('operator_name', '') == target_well.get('operator_name', '')).astype(int)
        
        # Vintage gap
        if 'first_prod_date' in candidates.columns and 'first_prod_date' in target_well:
            target_date = pd.to_datetime(target_well['first_prod_date'])
            candidate_dates = pd.to_datetime(candidates['first_prod_date'])
            vintage_gap = (target_date - candidate_dates).dt.days / 365.25
            features['vintage_gap_years'] = vintage_gap
            features['abs_vintage_gap'] = np.abs(vintage_gap)
        else:
            features['vintage_gap_years'] = 0
            features['abs_vintage_gap'] = 0
        
        # Add missing embedding features if needed
        if self.feature_columns:
            for col in self.feature_columns:
                if col.startswith('embedding_') and col not in features.columns:
                    features[col] = 0
        
        # Interaction features
        features['distance_x_formation'] = features['distance_mi'] * features['same_formation']
        features['distance_x_operator'] = features['distance_mi'] * features['same_operator']
        features['length_x_ppf'] = features['abs_log_length_ratio'] * features['log_ppf_ratio']
        
        # Ensure all columns are present and in correct order
        if self.feature_columns:
            features = features.reindex(columns=self.feature_columns, fill_value=0)
        
        # Score with model
        scores = self.model.predict(features, num_iteration=self.model.best_iteration)
        
        return scores
    
    def predict(self, target_well: Dict,
                return_analogs: bool = False,
                bootstrap_samples: Optional[int] = None) -> Dict:
        """
        Make production forecast for target well
        
        Args:
            target_well: Target well properties
            return_analogs: Whether to return analog details
            bootstrap_samples: Optional number of bootstrap samples for P10/P90
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Get candidates using baseline picker's method
        candidates = self.baseline_picker.find_candidates(target_well)
        
        if len(candidates) == 0:
            return {'error': 'No candidates found', 'n_analogs': 0}
        
        # Score with LightGBM model
        if self.model is not None:
            scores = self.score_candidates(candidates, target_well)
            candidates['score'] = scores
            candidates = candidates.sort_values('score', ascending=False)
        else:
            # Fall back to baseline scoring
            candidates = self.baseline_picker.score_candidates(target_well, candidates)
        
        # Select top K with operator diversity
        selected = self.baseline_picker.select_top_k(candidates, self.top_k)
        
        if len(selected) == 0:
            return {'error': 'No analogs selected', 'n_analogs': 0}
        
        # Warp and average (reuse baseline's logic)
        warped_curves = []
        
        for _, analog in selected.iterrows():
            oil_curve = np.array([float(v) for v in analog['oil_m1_9']])
            
            analog_props = {
                'lateral_length': analog['lateral_length'],
                'proppant_per_ft': analog.get('proppant_per_ft')
            }
            
            warped = self.baseline_picker.warp_production(oil_curve, analog_props, target_well)
            warped_curves.append(warped)
        
        warped_curves = np.array(warped_curves)
        
        # Calculate predictions
        p50_prediction = np.mean(warped_curves, axis=0)
        
        result = {
            'p50_oil_m1_9': p50_prediction.tolist(),
            'n_analogs': len(selected),
            'avg_distance_mi': selected['distance_mi'].mean(),
            'formation_match_pct': selected.get('formation_match', selected['formation'] == target_well['formation']).mean() * 100,
            'model_type': 'lightgbm'
        }
        
        # Bootstrap for P10/P90
        if bootstrap_samples and bootstrap_samples > 0 and len(warped_curves) > 1:
            bootstrap_predictions = []
            
            for _ in range(bootstrap_samples):
                idx = np.random.choice(len(warped_curves), size=len(warped_curves), replace=True)
                sample_mean = np.mean(warped_curves[idx], axis=0)
                bootstrap_predictions.append(sample_mean)
            
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            p10 = np.percentile(bootstrap_predictions, 10, axis=0)
            p90 = np.percentile(bootstrap_predictions, 90, axis=0)
            
            result['p10_oil_m1_9'] = p10.tolist()
            result['p90_oil_m1_9'] = p90.tolist()
        
        # Add analog details
        if return_analogs:
            result['analogs'] = selected[['well_id', 'operator_name', 'formation', 
                                         'distance_mi', 'score']].head(10).to_dict('records')
        
        return result
    
    def save(self, path: str):
        """Save model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dict = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'lgb_params': self.lgb_params,
            'top_k': self.top_k,
            'max_per_operator': self.max_per_operator,
            'kappa_length': self.kappa_length,
            'kappa_proppant': self.kappa_proppant
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and metadata"""
        save_dict = joblib.load(path)
        
        self.model = save_dict['model']
        self.feature_columns = save_dict['feature_columns']
        self.lgb_params = save_dict['lgb_params']
        self.top_k = save_dict['top_k']
        self.max_per_operator = save_dict['max_per_operator']
        self.kappa_length = save_dict['kappa_length']
        self.kappa_proppant = save_dict['kappa_proppant']
        
        logger.info(f"Model loaded from {path}")


def train_model(basin: Optional[str] = None, limit_wells: Optional[int] = None, save_path: Optional[str] = None):
    """
    Train and save the LightGBM ranker model
    
    Args:
        basin: Basin to train for ('dj_basin' or 'bakken')
        limit_wells: Optional limit on training wells
        save_path: Path to save trained model (defaults to models/lightgbm_ranker_{basin}.pkl)
    """
    if save_path is None:
        if basin:
            save_path = f'models/lightgbm_ranker_{basin}.pkl'
        else:
            save_path = 'models/lightgbm_ranker.pkl'
    
    logger.info(f"Starting LightGBM ranker training for {basin or 'all basins'}...")
    
    # Initialize model
    ranker = LightGBMAnalogRanker(basin=basin)
    
    # Prepare training data
    features, labels, groups = ranker.prepare_training_data(limit=limit_wells)
    
    # Split for validation (last 20% of targets)
    n_groups = len(groups)
    n_train_groups = int(0.8 * n_groups)
    
    train_size = sum(groups[:n_train_groups])
    
    train_features = features.iloc[:train_size]
    train_labels = labels[:train_size]
    train_groups = groups[:n_train_groups]
    
    valid_features = features.iloc[train_size:]
    valid_labels = labels[train_size:]
    valid_groups = groups[n_train_groups:]
    
    # Train model
    ranker.train(
        train_features, train_labels, train_groups,
        valid_features, valid_labels, valid_groups,
        num_boost_round=200,
        early_stopping_rounds=20
    )
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ranker.save(save_path)
    
    logger.info(f"Training complete. Model saved to {save_path}")
    
    return ranker


def train_basins_parallel(basins: List[str] = ['dj_basin', 'bakken'], limit_wells: Optional[int] = None):
    """
    Train models for multiple basins in parallel
    
    Args:
        basins: List of basins to train
        limit_wells: Optional limit on training wells per basin
    
    Returns:
        Dict of results {basin: (success, path_or_error)}
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    logger.info(f"Training {len(basins)} basins in parallel...")
    
    results = {}
    with ProcessPoolExecutor(max_workers=len(basins)) as executor:
        futures = {
            executor.submit(train_model, basin, limit_wells): basin 
            for basin in basins
        }
        
        for future in as_completed(futures):
            basin = futures[future]
            try:
                ranker = future.result()
                results[basin] = (True, f'models/lightgbm_ranker_{basin}.pkl')
                logger.info(f"✓ {basin} training completed")
            except Exception as e:
                results[basin] = (False, str(e))
                logger.error(f"✗ {basin} training failed: {str(e)}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LightGBM analog ranker")
    parser.add_argument('--basin', choices=['dj_basin', 'bakken'], help="Basin to train")
    parser.add_argument('--basins', nargs='+', default=None, help="Multiple basins to train in parallel")
    parser.add_argument('--limit', type=int, help="Limit number of training wells")
    parser.add_argument('--output', help="Output path for model")
    parser.add_argument('--parallel', action='store_true', help="Train basins in parallel")
    
    args = parser.parse_args()
    
    if args.basins and args.parallel:
        # Train multiple basins in parallel
        train_basins_parallel(args.basins, args.limit)
    elif args.basin:
        # Train single basin
        train_model(basin=args.basin, limit_wells=args.limit, save_path=args.output)
    else:
        # Train all data (no basin filter)
        train_model(limit_wells=args.limit, save_path=args.output)