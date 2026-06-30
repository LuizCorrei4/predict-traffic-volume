# Machine Learning Modeling Implementation Plan

This document details the architecture and workflow for the modeling phase in `src/models/`. The goal is to build a robust, modular, and extensible pipeline capable of training, tuning, and evaluating multiple Machine Learning algorithms for traffic volume prediction.

## 1. Directory Structure (`src/models/`)

To maintain clean and professional code, the modeling phase will be split into modular scripts, each with a single responsibility:

*   `src/models/data_splitter.py`: Handles loading the processed CSV, splitting it chronologically, and fitting/applying scalers strictly on the training set to prevent data leakage.
*   `src/models/algorithms.py`: Contains factory functions or classes to instantiate our chosen models (e.g., `get_xgboost()`, `get_linear_regression()`). This keeps our main script clean.
*   `src/models/evaluator.py`: Contains functions to compute evaluation metrics (RMSE, MAE, $R^2$) and plot predictions vs. reality (scatter plots and time-series line graphs).
*   `src/models/train_model.py`: The main orchestrator script (CLI tool). It ties the modules together, coordinates hyperparameter tuning, and saves the final best models.

## 2. Step-by-Step Implementation Workflow

### Phase 1: Data Splitting & Scaling (`data_splitter.py`)
1.  **Chronological Split (Holdout):** We will load `clean_traffic_data.csv` and sort it by `date_time`. We will *not* shuffle the data. We will slice it by index chronologically:
    *   **Train Set:** 70% of the timeline.
    *   **Validation Set:** 15% (Used for hyperparameter tuning and model selection).
    *   **Test Set:** 15% (Used for final, unbiased evaluation).
2.  **Feature Selection Integration:** Drop the redundant raw columns specified in our EDA (e.g., `weather_main`, `hour`) isolating only the engineered `X` (features) and `y` (`traffic_volume`).
3.  **Strict Scaling (Avoiding Data Leakage):**
    *   Instantiate a `StandardScaler`.
    *   Apply `fit_transform` **ONLY** on the continuous numerical columns of the **Train Set**.
    *   Apply `transform` (using the Train scaler) on the numerical columns of the **Validation** and **Test** sets.
4.  **Save Split Data:** To ensure reproducibility and facilitate potential separate modeling pipelines later, the final split datasets will be explicitly saved to `data/processed/` with semantically clear names (e.g., `X_train_scaled.csv`, `y_train.csv`, `X_val_scaled.csv`, `y_val.csv`, `X_test_scaled.csv`, `y_test.csv`).

### Phase 2: Algorithm Selection & Setup (`algorithms.py`)
We will set up three distinct algorithmic approaches to provide a comprehensive analysis for the study:
1.  **Linear Regression Baseline (Ridge/Lasso):** Essential for checking if the problem can be solved simply. Highly interpretable.
2.  **XGBoost Regressor (Tree-based):** State-of-the-art for tabular data. Handles non-linearities and complex feature interactions (like weather + rush hour) natively.
3.  **Multi-Layer Perceptron (Neural Network):** A basic deep learning approach (`sklearn.neural_network.MLPRegressor`) to capture highly complex mappings, testing if an MLP outperforms standard trees on this dataset.

### Phase 3: Hyperparameter Tuning Strategy
To find the best configuration for each model without manually guessing, we will use **RandomizedSearchCV** (or `GridSearchCV` for smaller spaces) combined with `TimeSeriesSplit`.
*   **TimeSeriesSplit:** A cross-validation technique in `scikit-learn` specifically for time series. Instead of random K-Folds, it validates on sequential forward-rolling chunks of the data, ensuring the model never sees future data during tuning.
*   **Workflow:** We will train on the Train set, tune hyperparameters using `TimeSeriesSplit`, validate against the Validation set to select the best model class, and finally test the absolute best model on the unseen Test set.

### Phase 4: Evaluation & Artifacts (`evaluator.py`)
For each trained model, the orchestrator will generate:
1.  **Metrics:** 
    *   **RMSE (Root Mean Squared Error):** Penalizes large errors (e.g., missing a huge traffic spike).
    *   **MAE (Mean Absolute Error):** The average error in absolute number of cars.
    *   **$R^2$ (R-Squared):** The proportion of the variance in traffic volume explained by the model.
2.  **Visualizations:** Scatter plots of Actual vs. Predicted, and Time Series line plots of a specific 1-week window to visually inspect if the model is accurately capturing the daily rush hour peaks.
3.  **Model Serialization:** The best performing model object and its fitted Scaler will be saved via `joblib` into the `models/` root directory for potential future deployment or API creation.

## 3. Why This Architecture? (For Machine Learning Engineering)

*   **Modularity:** If a researcher decides to test a new algorithm (e.g., LightGBM) tomorrow, they only need to add 5 lines of code in `algorithms.py` without touching the data splitting or evaluation logic.
*   **Reproducibility:** By centralizing the pipeline in `train_model.py` and controlling the random seeds, any user can clone the repository and reproduce the exact RMSE metrics reported in the final study.
*   **Leakage Prevention:** By strictly separating the scaler fitting process into the `data_splitter.py` module *after* the chronological split, we guarantee our validation and test metrics reflect true, real-world generalization.
