import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import time
import joblib
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.models.algorithms import get_algorithm, get_all_supported_algorithms
from src.models.evaluator import evaluate_model, save_experiment_log

def load_data(mode: str, data_dir: str = "data/processed"):
    """
    Loads the appropriate datasets based on the execution mode.
    Validation Mode: Loads Train for training, Val for evaluation.
    Test Mode: Loads Train+Val for training, Test for evaluation.
    """
    print(f"Loading data for mode: {mode.upper()}...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'), index_col='date_time')
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col='date_time')['traffic_volume']
    
    if mode == 'validation':
        X_eval = pd.read_csv(os.path.join(data_dir, 'X_val_scaled.csv'), index_col='date_time')
        y_eval = pd.read_csv(os.path.join(data_dir, 'y_val.csv'), index_col='date_time')['traffic_volume']
        return X_train, y_train, X_eval, y_eval
        
    elif mode == 'test':
        X_val = pd.read_csv(os.path.join(data_dir, 'X_val_scaled.csv'), index_col='date_time')
        y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'), index_col='date_time')['traffic_volume']
        
        # Merge Train and Val for final Test mode training
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])
        
        # Load Test set exclusively for evaluation
        X_test = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'), index_col='date_time')
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col='date_time')['traffic_volume']
        return X_train_full, y_train_full, X_test, y_test
    else:
        raise ValueError("Invalid mode. Use 'validation' or 'test'.")

def get_hyperparameter_space(trial, algorithm_name):
    """Defines the Optuna search space for each algorithm."""
    name = algorithm_name.lower().strip()
    
    if name == 'ridge':
        return {
            'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)
        }
    elif name == 'random_forest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
    elif name == 'xgboost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
    elif name == 'lightgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100)
        }
    elif name == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 100, 300, step=50),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
    elif name == 'mlp':
        return {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        }
    return {}

def objective(trial, algorithm_name, X_train, y_train):
    """
    Optuna objective function. Uses TimeSeriesSplit to evaluate hyperparams 
    without leaking future data. We use 3 splits to keep tuning fast.
    """
    params = get_hyperparameter_space(trial, algorithm_name)
    tscv = TimeSeriesSplit(n_splits=3)
    
    rmse_scores = []
    
    # Convert to numpy arrays for strict index-based slicing by TimeSeriesSplit
    X_arr = X_train.values
    y_arr = y_train.values
    
    for train_index, val_index in tscv.split(X_arr):
        X_fold_train, X_fold_val = X_arr[train_index], X_arr[val_index]
        y_fold_train, y_fold_val = y_arr[train_index], y_arr[val_index]
        
        model = get_algorithm(algorithm_name, **params)
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, preds))
        rmse_scores.append(fold_rmse)
        
    return np.mean(rmse_scores)

def run_experiment(algorithm_name, X_train, y_train, X_eval, y_eval, n_trials=10):
    """
    Orchestrates the tuning, training, and evaluation for a single algorithm.
    """
    print(f"\n{'='*50}\nStarting Experiment: {algorithm_name.upper()}\n{'='*50}")
    
    # 1. Hyperparameter Tuning via Optuna
    print(f"[{algorithm_name}] Running Optuna Optimization ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Silence excessive logs
    study = optuna.create_study(direction="minimize")
    
    # Pass arguments to the objective using a lambda
    study.optimize(lambda trial: objective(trial, algorithm_name, X_train, y_train), n_trials=n_trials)
    
    best_params = study.best_params
    print(f"[{algorithm_name}] Best Hyperparameters found: {best_params}")
    
    # 2. Final Training on the full training set (with best params)
    print(f"[{algorithm_name}] Training final model with best parameters...")
    start_time = time.time()
    
    final_model = get_algorithm(algorithm_name, **best_params)
    final_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # 3. Final Evaluation
    print(f"[{algorithm_name}] Evaluating model on the Holdout Set...")
    y_pred = final_model.predict(X_eval)
    metrics = evaluate_model(y_eval, y_pred)
    
    # 4. Save Logs and Model
    log_path = save_experiment_log(
        algorithm_name=algorithm_name,
        hyperparameters=best_params,
        metrics=metrics,
        training_time_seconds=training_time
    )
    
    # Save the model artifact
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{algorithm_name}_best.pkl"
    joblib.dump(final_model, model_path)
    print(f"[{algorithm_name}] Model serialized to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Machine Learning Orchestrator for Traffic Prediction")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=['validation', 'test'], 
        default='validation',
        help="Execution mode. 'validation' tunes on Train and evaluates on Val. 'test' trains on Train+Val and evaluates on Test."
    )
    parser.add_argument(
        "--models", 
        nargs='+', 
        default=get_all_supported_algorithms(),
        help="List of algorithms to run. E.g., --models xgboost lightgbm. Defaults to ALL."
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=10,
        help="Number of Optuna tuning trials per algorithm."
    )
    
    args = parser.parse_args()
    
    print(f"--- PREDICT TRAFFIC VOLUME - ML ORCHESTRATOR ---")
    print(f"Mode: {args.mode.upper()}")
    print(f"Models to run: {', '.join(args.models)}")
    print(f"Optuna Trials per model: {args.trials}")
    
    # Load Data
    X_train, y_train, X_eval, y_eval = load_data(args.mode)
    print(f"Training shape (Fit/CV): X={X_train.shape}, y={y_train.shape}")
    print(f"Evaluation shape (Final Scorer): X={X_eval.shape}, y={y_eval.shape}")
    
    # Run experiments sequentially
    for model_name in args.models:
        try:
            run_experiment(
                algorithm_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                n_trials=args.trials
            )
        except Exception as e:
            print(f"Error running experiment for {model_name}: {e}")

if __name__ == "__main__":
    main()
