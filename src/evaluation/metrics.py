"""Shared evaluation metrics for analog selection models"""

import numpy as np


def calculate_weighted_mae(actual, pred):
    """
    Calculate weighted Mean Absolute Error for months 1-9 of production.
    
    Earlier months are weighted more heavily to reflect their greater
    economic importance and prediction reliability.
    
    Args:
        actual: Array of actual monthly production values (at least 9 months)
        pred: Array of predicted monthly production values (at least 9 months)
    
    Returns:
        float: Weighted MAE value
    """
    # Weights for months 1-9 (earlier months count more)
    weights = [3, 3, 2, 2, 1.5, 1.5, 1, 1, 1]
    
    # Ensure we have 9 months of data
    actual_9m = actual[:9]
    pred_9m = pred[:9]
    
    # Calculate weighted MAE
    mae_components = [weights[i] * abs(actual_9m[i] - pred_9m[i]) for i in range(9)]
    weighted_mae = sum(mae_components) / sum(weights)
    
    return weighted_mae