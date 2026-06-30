import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.features_config import TARGET_FEATURE, FEATURES_TO_DROP, NUMERICAL_FEATURES_TO_SCALE
from src.models.features_config import TARGET_FEATURE, FEATURES_TO_DROP, NUMERICAL_FEATURES_TO_SCALE

def split_and_scale_data(
    input_filepath: str, 
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    """
    Reads the preprocessed data, applies feature selection, 
    performs a chronological holdout split, scales numerical features 
    strictly based on the train set, and saves the split datasets.
    """
    print(f"Loading data from {input_filepath}...")
    df = pd.read_csv(input_filepath, parse_dates=['date_time'], index_col='date_time')
    
    # 1. Feature Selection
    print("Dropping redundant and faulty columns based on features_config...")
    columns_to_drop = [col for col in FEATURES_TO_DROP if col in df.columns]
    df_filtered = df.drop(columns=columns_to_drop)
    
    X = df_filtered.drop(columns=[TARGET_FEATURE])
    y = df_filtered[TARGET_FEATURE]
    
    # 2. Chronological Split
    print("Performing chronological split (No Shuffling)...")
    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    print(f"Train set: {len(X_train)} rows")
    print(f"Validation set: {len(X_val)} rows")
    print(f"Test set: {len(X_test)} rows")
    
    # 3. Strict Scaling (Fit ONLY on Train to avoid leakage)
    print("Scaling continuous numerical features strictly based on Train Set...")
    scaler = StandardScaler()
    
    # Identify which columns to scale that are actually present
    cols_to_scale = [col for col in NUMERICAL_FEATURES_TO_SCALE if col in X_train.columns]
    
    # Create copies to avoid SettingWithCopyWarning
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    if cols_to_scale:
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        
    # 4. Save to Disk
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving split datasets to {output_dir}...")
    
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'))
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'))
    
    X_val_scaled.to_csv(os.path.join(output_dir, 'X_val_scaled.csv'))
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'))
    
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'))
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'))
    
    print("Data splitting and scaling completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and scale data chronologically.")
    parser.add_argument("--input", type=str, default="data/processed/clean_traffic_data.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    split_and_scale_data(args.input, args.output_dir)
