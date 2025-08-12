"""
Centralized feature engineering for LightGBM ranking model
Single source of truth for all feature generation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def build_ranking_features(
    candidates_df: pd.DataFrame,
    target_well: Dict,
    embeddings_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build standardized features for ranking model
    
    This is the SINGLE SOURCE OF TRUTH for feature engineering.
    Used by training, inference, and evaluation.
    
    Args:
        candidates_df: DataFrame with candidate wells containing:
            - distance_mi: Distance to target
            - lateral_length or candidate_lateral: Lateral length
            - proppant_per_ft or candidate_ppf: Proppant per foot
            - formation or candidate_formation: Formation name
            - operator_name or candidate_operator: Operator
            - first_prod_date or candidate_first_prod: First production date
            - For training: length_ratio, ppf_ratio, formation_match, etc may be pre-computed
        target_well: Dict with target well properties:
            - lateral_length: Target lateral length
            - proppant_per_ft: Target proppant per foot
            - formation: Target formation
            - operator_name: Target operator
            - first_prod_date: Target first production date
        embeddings_df: Optional DataFrame with curve embeddings
    
    Returns:
        DataFrame with standardized features
    """
    features = pd.DataFrame()
    
    # Standardize column names
    if 'candidate_lateral' in candidates_df.columns:
        candidates_df['lateral_length'] = candidates_df['candidate_lateral']
    if 'candidate_ppf' in candidates_df.columns:
        candidates_df['proppant_per_ft'] = candidates_df['candidate_ppf']
    if 'candidate_formation' in candidates_df.columns:
        candidates_df['formation'] = candidates_df['candidate_formation']
    if 'candidate_operator' in candidates_df.columns:
        candidates_df['operator_name'] = candidates_df['candidate_operator']
    if 'candidate_first_prod' in candidates_df.columns:
        candidates_df['first_prod_date'] = candidates_df['candidate_first_prod']
    
    # Distance features
    features['distance_mi'] = candidates_df['distance_mi']
    features['distance_km'] = candidates_df['distance_mi'] * 1.60934
    features['log_distance'] = np.log1p(candidates_df['distance_mi'])
    
    # Lateral length features
    target_lateral = float(target_well.get('lateral_length', 0))
    if target_lateral > 0:
        if 'length_ratio' in candidates_df.columns:
            features['length_ratio'] = candidates_df['length_ratio']
        else:
            features['length_ratio'] = candidates_df['lateral_length'] / target_lateral
        
        if 'delta_length' in candidates_df.columns:
            features['length_delta'] = candidates_df['delta_length']
        else:
            features['length_delta'] = candidates_df['lateral_length'] - target_lateral
    else:
        features['length_ratio'] = 1.0
        features['length_delta'] = 0.0
    
    features['log_length_ratio'] = np.log(features['length_ratio'].clip(lower=0.1))
    features['abs_log_length_ratio'] = np.abs(features['log_length_ratio'])
    
    # Proppant features
    target_ppf = float(target_well.get('proppant_per_ft', 0))
    if target_ppf > 0:
        if 'ppf_ratio' in candidates_df.columns:
            ppf_ratio = candidates_df['ppf_ratio']
        else:
            # Avoid division by zero
            ppf_ratio = target_ppf / (candidates_df['proppant_per_ft'].fillna(1) + 1)
        features['ppf_ratio'] = ppf_ratio
        features['log_ppf_ratio'] = np.log(ppf_ratio.clip(lower=0.1))
    else:
        features['ppf_ratio'] = 1.0
        features['log_ppf_ratio'] = 0.0
    
    # Formation matching
    # IMPORTANT: Should always be 1.0 if we're filtering by formation
    if 'formation_match' in candidates_df.columns:
        features['same_formation'] = candidates_df['formation_match'].astype(int)
    else:
        target_formation = target_well.get('formation', '')
        if target_formation:
            features['same_formation'] = (candidates_df['formation'] == target_formation).astype(int)
        else:
            features['same_formation'] = 1  # Assume match if no formation data
    
    # Operator matching
    if 'same_operator' in candidates_df.columns:
        features['same_operator'] = candidates_df['same_operator'].astype(int)
    else:
        target_operator = target_well.get('operator_name', '')
        if target_operator:
            features['same_operator'] = (candidates_df.get('operator_name', '') == target_operator).astype(int)
        else:
            features['same_operator'] = 0
    
    # Vintage gap
    if 'vintage_gap_years' in candidates_df.columns:
        features['vintage_gap_years'] = candidates_df['vintage_gap_years'].fillna(0)
    else:
        # Calculate from dates
        target_date = pd.to_datetime(target_well.get('first_prod_date'))
        if target_date and 'first_prod_date' in candidates_df.columns:
            candidate_dates = pd.to_datetime(candidates_df['first_prod_date'])
            vintage_gap = (target_date - candidate_dates).dt.days / 365.25
            features['vintage_gap_years'] = vintage_gap.fillna(0)
        else:
            features['vintage_gap_years'] = 0
    
    features['abs_vintage_gap'] = np.abs(features['vintage_gap_years'])
    
    # Embeddings (if available)
    if embeddings_df is not None and 'candidate_embedding' in candidates_df.columns:
        # Parse embedding arrays
        embeddings = np.vstack(candidates_df['candidate_embedding'].apply(
            lambda x: np.array(x) if x is not None else np.zeros(4)
        ))
        for i in range(min(4, embeddings.shape[1])):
            features[f'embedding_pc{i+1}'] = embeddings[:, i]
    else:
        # Add zero embeddings if expected
        for i in range(4):
            features[f'embedding_pc{i+1}'] = 0.0
    
    # Interaction features
    features['distance_x_formation'] = features['distance_mi'] * features['same_formation']
    features['distance_x_operator'] = features['distance_mi'] * features['same_operator']
    features['length_x_ppf'] = features['abs_log_length_ratio'] * features['log_ppf_ratio']
    
    # Ensure consistent feature order
    expected_features = [
        'distance_mi', 'distance_km', 'log_distance',
        'length_ratio', 'length_delta', 'log_length_ratio', 'abs_log_length_ratio',
        'ppf_ratio', 'log_ppf_ratio',
        'same_formation', 'same_operator',
        'vintage_gap_years', 'abs_vintage_gap',
        'embedding_pc1', 'embedding_pc2', 'embedding_pc3', 'embedding_pc4',
        'distance_x_formation', 'distance_x_operator', 'length_x_ppf'
    ]
    
    # Reorder and ensure all features exist
    for feat in expected_features:
        if feat not in features.columns:
            features[feat] = 0.0
            logger.warning(f"Feature {feat} not found, filling with zeros")
    
    features = features[expected_features]
    
    return features


def get_feature_names() -> list:
    """
    Get the list of feature names in the correct order
    
    Returns:
        List of feature names
    """
    return [
        'distance_mi', 'distance_km', 'log_distance',
        'length_ratio', 'length_delta', 'log_length_ratio', 'abs_log_length_ratio',
        'ppf_ratio', 'log_ppf_ratio',
        'same_formation', 'same_operator',
        'vintage_gap_years', 'abs_vintage_gap',
        'embedding_pc1', 'embedding_pc2', 'embedding_pc3', 'embedding_pc4',
        'distance_x_formation', 'distance_x_operator', 'length_x_ppf'
    ]