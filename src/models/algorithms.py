"""
Factory module for initializing Machine Learning algorithms.
Centralizing model instantiation here keeps the main scripts clean 
and makes it easy to add new algorithms in the future.
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def get_algorithm(algorithm_name: str, **kwargs):
    """
    Factory function to return an un-fitted model instance.
    
    Args:
        algorithm_name (str): The name of the algorithm.
        **kwargs: Optional hyperparameters to override defaults.
        
    Returns:
        model: An instantiated scikit-learn compatible regressor.
    """
    name = algorithm_name.lower().strip()
    
    # We set random_state=42 for reproducibility where applicable.
    
    if name == 'ridge':
        return Ridge(random_state=42, **kwargs)
        
    elif name == 'random_forest':
        # n_jobs=-1 uses all CPU cores
        return RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs)
        
    elif name == 'xgboost':
        return xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror', **kwargs)
        
    elif name == 'lightgbm':
        # verbose=-1 suppresses LightGBM's excessive logging
        return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **kwargs)
        
    elif name == 'catboost':
        # verbose=False suppresses iterative output during tuning
        return cb.CatBoostRegressor(random_seed=42, thread_count=-1, verbose=False, **kwargs)
        
    elif name == 'mlp':
        # Multi-Layer Perceptron (Neural Network)
        # We increase max_iter since default 200 is often too low for convergence
        return MLPRegressor(random_state=42, max_iter=1000, **kwargs)
        
    else:
        raise ValueError(f"Algorithm '{algorithm_name}' is not supported. "
                         f"Supported algorithms: ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'mlp']")

def get_all_supported_algorithms():
    """Returns a list of all algorithm names supported by this factory."""
    return ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'mlp']
