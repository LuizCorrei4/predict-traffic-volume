# Machine Learning Modeling Implementation Plan

This document details the architecture and workflow for the modeling phase in `src/models/`. The goal is to build a robust, modular, and extensible pipeline capable of training, tuning, and evaluating multiple state-of-the-art Machine Learning algorithms for traffic volume prediction.

## 1. Directory Structure (`src/models/`)

To maintain clean and professional code, the modeling phase will be split into modular scripts, each with a single responsibility:

*   `src/models/data_splitter.py`: Handles loading the processed CSV, splitting it chronologically, and fitting/applying scalers strictly on the training set to prevent data leakage.
*   `src/models/algorithms.py`: Contains factory functions or classes to instantiate our chosen models (e.g., `get_xgboost()`, `get_lightgbm()`, `get_catboost()`). This keeps our main script clean.
*   `src/models/evaluator.py`: Contains functions to compute evaluation metrics (RMSE, MAE, $R^2$), measure training time, plot predictions vs. reality, and log experimental results.
*   `src/models/train_model.py`: The main orchestrator script (CLI tool). It ties the modules together, coordinates hyperparameter tuning with Optuna, and manages the training/evaluation lifecycle via command-line arguments.

## 2. Step-by-Step Implementation Workflow

### Phase 1: Data Splitting & Scaling (`data_splitter.py`)
1.  **Chronological Split (Holdout):** We will load `clean_traffic_data.csv` and sort it by `date_time`. We will *not* shuffle the data. We will slice it by index chronologically:
    *   **Train Set:** 70% of the timeline.
    *   **Validation (Dev) Set:** 15% (Used for hyperparameter tuning and model selection).
    *   **Test Set:** 15% (Used exclusively for final, unbiased evaluation).
2.  **Modular Feature Selection Integration:** To avoid hardcoding column names inside the splitting logic, we will create a configuration structure (e.g., `src/models/features_config.json` or `.py`). This config will explicitly list the columns to KEEP and DROP exactly as concluded in `notebooks/02-eda_after_preprocess_insights.md` (e.g., dropping `weather_main`, `hour`, etc.). The `data_splitter.py` will dynamically read this configuration to filter the dataset, isolating `X` (features) and `y` (`traffic_volume`). This guarantees the code remains agnostic, clean, and highly maintainable.
3.  **Strict Scaling (Avoiding Data Leakage):**
    *   Apply `fit_transform` of `StandardScaler` **ONLY** on the continuous numerical columns of the **Train Set**.
    *   Apply `transform` on the **Validation** and **Test** sets.
4.  **Save Split Data:** The final split datasets will be explicitly saved to `data/processed/` with semantically clear names (e.g., `X_train_scaled.csv`, `y_train.csv`, `X_val_scaled.csv`, etc.).

### Phase 2: Algorithm Selection & Setup (`algorithms.py`)
To ensure our study evaluates the true state-of-the-art for tabular regression, we will implement a robust suite of algorithms:
1.  **Linear Regression (Ridge/Lasso):** The classical baseline. Essential for checking if the problem can be solved with simple linear combinations.
2.  **Random Forest (Bagging):** Highly robust to noise and outliers. Serves as a powerful non-linear baseline.
3.  **The "Boosting" Trio:** The current absolute state-of-the-art for tabular data.
    *   **XGBoost:** The industry standard gradient booster. Excellent at complex feature interactions.
    *   **LightGBM (Microsoft):** Often significantly faster to train than XGBoost and highly memory-efficient, making it ideal for tracking training-time efficiency.
    *   **CatBoost (Yandex):** Utilizes symmetric (oblivious) trees, making it extremely resistant to overfitting. Often yields the best out-of-the-box performance.
4.  **Multi-Layer Perceptron (Neural Network):** A basic deep learning approach to test if a neural architecture can outperform tree-based methods on cyclical temporal features.

### Phase 3: Advanced Hyperparameter Tuning & Execution Modes
We will bypass simple grid searches and adopt **Bayesian Optimization** using the `Optuna` library to find optimal configurations efficiently. 

**Execution Modes via CLI Arguments:**
The orchestrator script (`train_model.py`) will be engineered to support two distinct execution modes via command-line arguments (e.g., `--mode validation` vs `--mode test`):

*   **Mode 1: Validation & Tuning (`--mode validation`)**
    *   **Action:** Trains the algorithms on the `Train` set and uses `Optuna` (with `TimeSeriesSplit`) to evaluate performance on the `Validation` set.
    *   **The Golden Rule:** Under absolutely no circumstances will the Test Set be loaded or touched during this mode.
*   **Mode 2: Final Testing (`--mode test`)**
    *   **Action:** Once the hyperparameters are finalized from Mode 1, this mode dynamically merges the Train and Validation sets together (increasing the total historical data). It trains the champion models on `Train + Val`, and evaluates them **strictly once** on the `Test` set to simulate real-world deployment.

### Phase 4: Comprehensive Evaluation & Logging (`evaluator.py`)
To enrich the final academic report, the evaluation script will meticulously log every experimental run. Instead of a single messy file, logs will be saved as individual JSON/CSV files inside a `reports/logs/` directory.

**Semantic Filenaming Convention:** 
Each log file will have a highly semantic and chronological name containing the algorithm and the exact timestamp (Year, Month, Day, Hour, Minute). 
*Format:* `[Algorithm]_[YYYY-MM-DD_HH-MM]_log.json` 
*(Example: `LightGBM_2026-06-30_19-15_log.json`)*

**Metrics Logged per Run:**
1.  **Algorithm Details:** 
    *   Algorithm Name (e.g., "LightGBM").
    *   Underlying Library/Tool (e.g., "lightgbm library v4.3").
    *   Exact Hyperparameters used (e.g., `learning_rate=0.01`, `max_depth=6`).
2.  **Performance Metrics:** RMSE, MAE, and $R^2$.
3.  **Computational Cost:** **Training Time (in seconds)**. This is crucial for comparing the trade-off between predictive accuracy and computational efficiency (e.g., comparing XGBoost speed vs. LightGBM speed in the final report).

**Visual Artifacts:** Scatter plots of Actual vs. Predicted, and Time Series line plots of a 1-week window to inspect peak capture. Best models will be serialized via `joblib`.
