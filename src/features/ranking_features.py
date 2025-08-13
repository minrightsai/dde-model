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
    peak_features_df: Optional[pd.DataFrame] = None
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
            - wells_within_1mi_at_start: Wells within 1 mile when candidate started
            - wells_within_3mi_at_start: Wells within 3 miles when candidate started
            - distance_to_nearest_at_start: Distance to nearest well when candidate started
            - distance_to_second_nearest_at_start: Distance to 2nd nearest when candidate started
            - For training: length_ratio, ppf_ratio, formation_match, etc may be pre-computed
        target_well: Dict with target well properties:
            - lateral_length: Target lateral length
            - proppant_per_ft: Target proppant per foot
            - formation: Target formation
            - operator_name: Target operator
            - first_prod_date: Target first production date
            - peak_oil: Target peak oil (if available)
            - oil_decline_rate: Target decline rate (if available)
        peak_features_df: Optional DataFrame with peak production features
    
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
    
    # Peak production features (replacing embeddings)
    if peak_features_df is not None:
        # Merge peak features if available
        if 'candidate_id' in candidates_df.columns:
            peak_data = peak_features_df[peak_features_df['well_id'].isin(candidates_df['candidate_id'])]
            peak_data = peak_data.set_index('well_id').reindex(candidates_df['candidate_id']).reset_index(drop=True)
        elif 'well_id' in candidates_df.columns:
            peak_data = peak_features_df[peak_features_df['well_id'].isin(candidates_df['well_id'])]
            peak_data = peak_data.set_index('well_id').reindex(candidates_df['well_id']).reset_index(drop=True)
        else:
            peak_data = pd.DataFrame()
        
        if not peak_data.empty:
            features['candidate_peak_oil'] = peak_data['peak_oil'].fillna(0)
            features['candidate_peak_gas'] = peak_data['peak_gas'].fillna(0)
            
            # Peak ratios if target has peak data
            target_peak_oil = target_well.get('peak_oil', 0)
            if target_peak_oil > 0:
                features['peak_oil_ratio'] = features['candidate_peak_oil'] / (target_peak_oil + 1)
                features['log_peak_oil_ratio'] = np.log(features['peak_oil_ratio'].clip(lower=0.1))
            else:
                features['peak_oil_ratio'] = 1.0
                features['log_peak_oil_ratio'] = 0.0
            
            features['peak_month_diff'] = peak_data['peak_oil_month'].fillna(3) - target_well.get('peak_oil_month', 3)
            features['early_peak'] = (peak_data['peak_oil_month'] <= 3).astype(int).fillna(1)
            
            # Decline rate difference
            target_decline = target_well.get('oil_decline_rate', 0.7)
            features['decline_rate_diff'] = peak_data['oil_decline_rate'].fillna(0.7) - target_decline
        else:
            # Default values if no peak data
            features['candidate_peak_oil'] = 0
            features['candidate_peak_gas'] = 0
            features['peak_oil_ratio'] = 1.0
            features['log_peak_oil_ratio'] = 0.0
            features['peak_month_diff'] = 0
            features['early_peak'] = 1
            features['decline_rate_diff'] = 0
    else:
        # Add zero peak features if not available
        features['candidate_peak_oil'] = 0
        features['candidate_peak_gas'] = 0
        features['peak_oil_ratio'] = 1.0
        features['log_peak_oil_ratio'] = 0.0
        features['peak_month_diff'] = 0
        features['early_peak'] = 1
        features['decline_rate_diff'] = 0
    
    # Time-aware well spacing features
    if 'wells_within_1mi_at_start' in candidates_df.columns:
        features['wells_within_1mi_at_start'] = candidates_df['wells_within_1mi_at_start'].fillna(0)
    else:
        features['wells_within_1mi_at_start'] = 0
    
    if 'wells_within_3mi_at_start' in candidates_df.columns:
        features['wells_within_3mi_at_start'] = candidates_df['wells_within_3mi_at_start'].fillna(0)
    else:
        features['wells_within_3mi_at_start'] = 0
    
    if 'distance_to_nearest_at_start' in candidates_df.columns:
        features['distance_to_nearest_at_start'] = candidates_df['distance_to_nearest_at_start'].fillna(999)
    else:
        features['distance_to_nearest_at_start'] = 999
    
    if 'distance_to_second_nearest_at_start' in candidates_df.columns:
        features['distance_to_second_nearest_at_start'] = candidates_df['distance_to_second_nearest_at_start'].fillna(999)
    else:
        features['distance_to_second_nearest_at_start'] = 999
    
    # Interaction features
    features['distance_x_formation'] = features['distance_mi'] * features['same_formation']
    features['distance_x_operator'] = features['distance_mi'] * features['same_operator']
    features['length_x_ppf'] = features['abs_log_length_ratio'] * features['log_ppf_ratio']
    
    # Ensure consistent feature order (17 features as per MODEL_TWEAKS.md)
    expected_features = [
        # Distance (2)
        'distance_mi', 'log_distance',
        # Lateral Length (3)
        'length_delta', 'log_length_ratio', 'abs_log_length_ratio',
        # Completion Design (2)
        'ppf_ratio', 'log_ppf_ratio',
        # Operator (1)
        'same_operator',
        # Temporal (1)
        'vintage_gap_years',
        # Peak Production (4)
        'candidate_peak_oil', 'peak_oil_ratio', 'log_peak_oil_ratio', 'decline_rate_diff',
        # Time-Aware Well Spacing (4)
        'wells_within_1mi_at_start', 'wells_within_3mi_at_start', 
        'distance_to_nearest_at_start', 'distance_to_second_nearest_at_start'
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
    Get the list of feature names in the correct order (17 features)
    
    Returns:
        List of feature names
    """
    return [
        # Distance (2)
        'distance_mi', 'log_distance',
        # Lateral Length (3)
        'length_delta', 'log_length_ratio', 'abs_log_length_ratio',
        # Completion Design (2)
        'ppf_ratio', 'log_ppf_ratio',
        # Operator (1)
        'same_operator',
        # Temporal (1)
        'vintage_gap_years',
        # Peak Production (4)
        'candidate_peak_oil', 'peak_oil_ratio', 'log_peak_oil_ratio', 'decline_rate_diff',
        # Time-Aware Well Spacing (4)
        'wells_within_1mi_at_start', 'wells_within_3mi_at_start', 
        'distance_to_nearest_at_start', 'distance_to_second_nearest_at_start'
    ]