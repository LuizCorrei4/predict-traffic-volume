# Source Directory (`src/`)

This directory contains the modular codebase for the `predict-traffic-volume` project.

## Structure
*   **`data/`**: Data loading and preprocessing pipelines.
    *   `preprocessing.py`: Implements raw data cleaning, outlier treatment, missing data filtering, and linear/ffill gap imputation.
*   **`features/`**: Feature engineering logic.
    *   `build_features.py`: Computes cyclical time transformations, binary weather indicators, weekend/rush-hour flags, and OHE/numerical scaling.
*   **`run_pipeline.py`**: End-to-end command-line runner that executes the full data preprocessing and feature engineering pipeline.

---

## Commands and Usage

We use `taskipy` (managed via Poetry) to run commands. Make sure you have activated the environment with `poetry shell` before running these.

### 1. Run Preprocessing & Feature Engineering Pipeline
This executes the full pipeline, loads the raw dataset, processes it, constructs features, drops targets containing NaNs (large gaps), and saves the outputs.
```bash
poetry run task preprocess
```
*Or if you are inside `poetry shell`:*
```bash
task preprocess
```

**Alternative with arguments:**
You can run the script directly with custom arguments:
```bash
python src/run_pipeline.py --raw data/raw/Metro_Interstate_Traffic_Volume.csv --processed data/processed/clean_traffic_data.csv --scalers models/scalers_and_encoders.pkl
```

### 2. Run Tests
Execute the unit tests in the repository:
```bash
poetry run task test
```
*Or if you are inside `poetry shell`:*
```bash
task test
```

### 3. Format Code
Automatically formats the code using `black` and sorts imports with `isort`:
```bash
poetry run task format
```
*Or if you are inside `poetry shell`:*
```bash
task format
```
