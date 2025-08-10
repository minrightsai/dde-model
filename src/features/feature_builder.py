"""
Step 5: Feature Engineering for ML Analog Scorer
Builds features for XGBoost model to learn analog selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.db_connector import DatabaseConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build features for ML analog scoring"""
    
    def __init__(self):
        """Initialize feature builder"""
        self.db = DatabaseConnector()
        self.feature_columns = []
    
    def load_training_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load training data from analog_candidates and early_rates
        
        Args:
            limit: Optional limit on number of target wells to load
            
        Returns:
            DataFrame with candidate pairs and features
        """
        query = """
        WITH training_targets AS (
            SELECT DISTINCT target_well_id
            FROM data.analog_candidates
            WHERE target_well_id IN (
                SELECT well_id 
                FROM data.early_rates 
                WHERE EXTRACT(YEAR FROM first_prod_date) < 2020  -- Pre-2020 for training
            )
            ORDER BY target_well_id
            {limit_clause}
        )
        SELECT 
            -- IDs
            ac.target_well_id,
            ac.candidate_well_id,
            
            -- Target well features
            t.formation as target_formation,
            t.operator_name as target_operator,
            t.lateral_length as target_lateral_length,
            t.proppant_per_ft as target_proppant_per_ft,
            t.fluid_per_ft as target_fluid_per_ft,
            EXTRACT(YEAR FROM t.first_prod_date) as target_vintage_year,
            EXTRACT(MONTH FROM t.first_prod_date) as target_vintage_month,
            t.latitude as target_lat,
            t.longitude as target_lon,
            t.cum_oil_m1_9 as target_cum_oil,
            t.cum_gas_m1_9 as target_cum_gas,
            t.avg_oil_m1_9 as target_avg_oil,
            t.oil_m1_9 as target_oil_curve,
            
            -- Candidate well features  
            c.formation as candidate_formation,
            c.operator_name as candidate_operator,
            c.lateral_length as candidate_lateral_length,
            c.proppant_per_ft as candidate_proppant_per_ft,
            c.fluid_per_ft as candidate_fluid_per_ft,
            EXTRACT(YEAR FROM c.first_prod_date) as candidate_vintage_year,
            EXTRACT(MONTH FROM c.first_prod_date) as candidate_vintage_month,
            c.latitude as candidate_lat,
            c.longitude as candidate_lon,
            c.cum_oil_m1_9 as candidate_cum_oil,
            c.cum_gas_m1_9 as candidate_cum_gas,
            c.avg_oil_m1_9 as candidate_avg_oil,
            c.oil_m1_9 as candidate_oil_curve,
            
            -- Pre-computed deltas from analog_candidates
            ac.distance_mi as distance_miles,
            ac.length_ratio as lateral_length_ratio,
            ac.delta_length as lateral_length_delta,
            ac.formation_match,
            ac.same_operator,
            ac.vintage_gap_years,
            ac.ppf_ratio as proppant_ratio_precomputed,
            ac.fpf_ratio as fluid_ratio_precomputed,
            
            -- Embeddings
            te.oil_embedding as target_embedding,
            ce.oil_embedding as candidate_embedding
            
        FROM data.analog_candidates ac
        JOIN training_targets tt ON ac.target_well_id = tt.target_well_id
        JOIN data.early_rates t ON ac.target_well_id = t.well_id
        JOIN data.early_rates c ON ac.candidate_well_id = c.well_id
        LEFT JOIN data.curve_embeddings te ON t.well_id = te.well_id
        LEFT JOIN data.curve_embeddings ce ON c.well_id = ce.well_id
        WHERE ac.distance_mi <= 15  -- Focus on local analogs
        ORDER BY ac.target_well_id, ac.distance_mi
        """
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = query.format(limit_clause=limit_clause)
        
        logger.info(f"Loading training data{f' (limit {limit} wells)' if limit else ''}...")
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} candidate pairs for {df['target_well_id'].nunique()} target wells")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from raw data
        
        Args:
            df: Raw dataframe from load_training_data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        # Convert arrays to floats
        for col in ['target_oil_curve', 'candidate_oil_curve']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: [float(v) for v in x] if x is not None else None)
        
        # Design parameter ratios
        df['proppant_ratio'] = df['candidate_proppant_per_ft'] / (df['target_proppant_per_ft'] + 1)
        df['fluid_ratio'] = df['candidate_fluid_per_ft'] / (df['target_fluid_per_ft'] + 1)
        
        # Log ratios for better scaling
        df['log_lateral_ratio'] = np.log1p(df['lateral_length_ratio'])
        df['log_proppant_ratio'] = np.log1p(df['proppant_ratio'])
        df['log_fluid_ratio'] = np.log1p(df['fluid_ratio'])
        
        # Production intensity
        df['target_oil_per_ft'] = df['target_cum_oil'] / (df['target_lateral_length'] + 1)
        df['candidate_oil_per_ft'] = df['candidate_cum_oil'] / (df['candidate_lateral_length'] + 1)
        df['oil_per_ft_ratio'] = df['candidate_oil_per_ft'] / (df['target_oil_per_ft'] + 1)
        
        # Vintage features
        df['vintage_match'] = (df['target_vintage_year'] == df['candidate_vintage_year']).astype(int)
        df['vintage_gap_abs'] = np.abs(df['vintage_gap_years'])
        
        # Seasonal alignment
        df['month_diff'] = np.abs(df['target_vintage_month'] - df['candidate_vintage_month'])
        df['same_season'] = (df['month_diff'] <= 2).astype(int)
        
        # Distance buckets
        df['distance_bucket'] = pd.cut(df['distance_miles'], 
                                       bins=[0, 1, 3, 5, 10, 15], 
                                       labels=['0-1mi', '1-3mi', '3-5mi', '5-10mi', '10-15mi'])
        
        # Embedding similarity (if available)
        if 'target_embedding' in df.columns and 'candidate_embedding' in df.columns:
            df = self._compute_embedding_similarity(df)
        
        # Operator categories
        major_operators = df['candidate_operator'].value_counts().head(10).index
        df['is_major_operator'] = df['candidate_operator'].isin(major_operators).astype(int)
        
        # Formation grouping (if certain formations are similar)
        df['formation_group'] = df['target_formation']  # Could map to groups if needed
        
        logger.info(f"Engineered {len(df.columns)} total features")
        
        return df
    
    def _compute_embedding_similarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cosine similarity between target and candidate embeddings"""
        
        def cosine_similarity(a, b):
            if a is None or b is None:
                return 0
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
        df['embedding_similarity'] = df.apply(
            lambda row: cosine_similarity(row['target_embedding'], row['candidate_embedding']),
            axis=1
        )
        
        # Also compute individual component differences
        for i in range(4):  # Assuming 4 PCA components
            df[f'embedding_diff_{i}'] = df.apply(
                lambda row: float(row['candidate_embedding'][i] - row['target_embedding'][i]) 
                if row['target_embedding'] is not None and row['candidate_embedding'] is not None 
                else 0,
                axis=1
            )
        
        return df
    
    def create_training_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create training labels (how good is each analog)
        
        The label is based on how close the candidate's production is to the target's
        Lower error = better analog = higher score
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with labels added
        """
        logger.info("Creating training labels...")
        
        def compute_production_error(target_curve, candidate_curve):
            """Compute normalized error between curves"""
            if target_curve is None or candidate_curve is None:
                return 1.0
            
            target = np.array(target_curve)
            candidate = np.array(candidate_curve)
            
            # Revenue-weighted MAE (early months matter more)
            weights = np.array([1.0 / (1.01 ** i) for i in range(9)])
            weights = weights / weights.sum()
            
            mae = np.average(np.abs(target - candidate), weights=weights)
            # Normalize by target average
            normalized_error = mae / (np.mean(target) + 1)
            
            return normalized_error
        
        # Compute error for each candidate
        df['production_error'] = df.apply(
            lambda row: compute_production_error(row['target_oil_curve'], row['candidate_oil_curve']),
            axis=1
        )
        
        # Convert error to score (lower error = higher score)
        # Using exponential to emphasize good matches
        df['analog_score'] = np.exp(-df['production_error'])
        
        # Also create binary labels for classification approach
        # Good analog = error < 20%
        df['is_good_analog'] = (df['production_error'] < 0.2).astype(int)
        
        logger.info(f"Created labels - {df['is_good_analog'].mean():.1%} are good analogs")
        
        return df
    
    def prepare_for_xgboost(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Prepare data for XGBoost training
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Tuple of (features, labels, groups, feature_names)
        """
        # Features to use for training
        feature_cols = [
            # Distance and location
            'distance_miles',
            'distance_bucket',
            
            # Lateral length
            'lateral_length_ratio',
            'log_lateral_ratio',
            'lateral_length_delta',
            
            # Design parameters
            'proppant_ratio',
            'fluid_ratio',
            'log_proppant_ratio',
            'log_fluid_ratio',
            
            # Formation and operator
            'formation_match',
            'same_operator',
            'is_major_operator',
            
            # Vintage
            'vintage_gap_years',
            'vintage_gap_abs',
            'vintage_match',
            'same_season',
            'month_diff',
            
            # Production intensity
            'candidate_oil_per_ft',
            'oil_per_ft_ratio',
            
            # Raw values for context
            'target_lateral_length',
            'candidate_lateral_length',
            'target_proppant_per_ft',
            'candidate_proppant_per_ft',
            'target_vintage_year',
            'candidate_vintage_year',
        ]
        
        # Add embedding features if available
        if 'embedding_similarity' in df.columns:
            feature_cols.append('embedding_similarity')
            feature_cols.extend([f'embedding_diff_{i}' for i in range(4)])
        
        # Handle categorical features
        categorical_features = ['distance_bucket']
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)
        
        # Update feature columns to include encoded columns
        encoded_cols = [col for col in df_encoded.columns if any(cat in col for cat in categorical_features)]
        feature_cols = [col for col in feature_cols if col not in categorical_features] + encoded_cols
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df_encoded.columns]
        
        X = df_encoded[feature_cols]
        y = df_encoded['analog_score']  # Or 'is_good_analog' for classification
        
        # Groups for ranking (all candidates for same target well)
        groups = df_encoded.groupby('target_well_id').size().values
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Target wells: {len(groups)}, avg candidates per well: {np.mean(groups):.1f}")
        
        self.feature_columns = feature_cols
        
        return X, y, groups, feature_cols
    
    def save_training_data(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray, 
                          output_dir: str = 'data/ml_features'):
        """Save prepared training data"""
        os.makedirs(output_dir, exist_ok=True)
        
        X.to_parquet(f'{output_dir}/X_train.parquet')
        y.to_frame('analog_score').to_parquet(f'{output_dir}/y_train.parquet')
        np.save(f'{output_dir}/groups_train.npy', groups)
        
        # Save feature names
        with open(f'{output_dir}/feature_names.txt', 'w') as f:
            for feat in self.feature_columns:
                f.write(f"{feat}\n")
        
        logger.info(f"Saved training data to {output_dir}")
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float], 
                                      output_path: str = 'feature_importance.png'):
        """Create feature importance visualization"""
        import matplotlib.pyplot as plt
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features for Analog Selection')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved feature importance plot to {output_path}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build features for ML analog scorer')
    parser.add_argument('--limit', type=int, help='Limit number of target wells')
    parser.add_argument('--save', action='store_true', help='Save training data to disk')
    args = parser.parse_args()
    
    builder = FeatureBuilder()
    
    # Load and process data
    df = builder.load_training_data(limit=args.limit)
    df = builder.engineer_features(df)
    df = builder.create_training_labels(df)
    
    # Prepare for XGBoost
    X, y, groups, feature_names = builder.prepare_for_xgboost(df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total samples: {len(X):,}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Target wells: {len(groups):,}")
    print(f"Avg candidates per well: {np.mean(groups):.1f}")
    print(f"Good analogs: {(y > 0.8).mean():.1%}")
    print(f"\nTop 10 features:")
    for i, feat in enumerate(feature_names[:10], 1):
        print(f"  {i}. {feat}")
    print("="*60)
    
    if args.save:
        builder.save_training_data(X, y, groups)
        
        # Also save the full dataframe for analysis
        df.to_parquet('data/ml_features/full_training_data.parquet')
        print(f"\nSaved training data to data/ml_features/")


if __name__ == "__main__":
    main()