"""
Module for evaluating machine learning models and logging experimental results.
Generates academic-standard metrics (RMSE, MAE, R2, MAPE, Max Error) 
and saves them as semantic, timestamped JSON files.
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes MAPE. Avoids division by zero by ignoring exact zeros 
    (though traffic volume is rarely strictly zero in the dataset).
    Returns the percentage value (e.g. 12.5 means 12.5%).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Prevent division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0 # Edge case if all targets are literally zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(y_true, y_pred):
    """
    Calculates the 5 essential regression metrics requested in the plan.
    
    Returns:
        dict: A dictionary containing RMSE, MAE, R2, MAPE, and Max Error.
    """
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_percentage": float(mean_absolute_percentage_error(y_true, y_pred)),
        "Max_Error": float(max_error(y_true, y_pred))
    }
    return metrics

def save_experiment_log(
    algorithm_name: str,
    hyperparameters: dict,
    metrics: dict,
    training_time_seconds: float,
    reports_dir: str = "reports/logs"
):
    """
    Saves the experimental run into a semantic JSON file.
    Format: reports/logs/[Algorithm]_[YYYY-MM-DD_HH-MM]_log.json
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create the semantic timestamp
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
    filename = f"{algorithm_name}_{timestamp_str}_log.json"
    filepath = os.path.join(reports_dir, filename)
    
    # Prepare the payload
    log_data = {
        "timestamp_exact": now.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": algorithm_name,
        "training_time_seconds": round(training_time_seconds, 2),
        "metrics": metrics,
        "hyperparameters": hyperparameters
    }
    
    # In case multiple runs of the same algorithm happen in the exact same minute
    # (e.g., during rapid hyperparameter tuning iterations), we append an index
    counter = 1
    while os.path.exists(filepath):
        filename = f"{algorithm_name}_{timestamp_str}_{counter}_log.json"
        filepath = os.path.join(reports_dir, filename)
        counter += 1
        
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    print(f"[{algorithm_name}] Experiment logged: {filepath} | RMSE: {metrics['RMSE']:.2f}")
    return filepath
