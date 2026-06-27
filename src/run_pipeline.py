import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.data.preprocessing import preprocess_pipeline
from src.features.build_features import pipeline_feature_engineering

def run(raw_path: str, processed_path: str):
    print("Starting preprocessing and feature engineering pipeline...")
    
    # 1. Preprocessing
    print(f"Loading and preprocessing raw data from {raw_path}...")
    df_preprocessed = preprocess_pipeline(raw_path)
    print(f"Preprocessed shape: {df_preprocessed.shape}")
    
    # 2. Feature Engineering
    print("Engineering features, generating cyclical time variables and weather indicators...")
    df_features = pipeline_feature_engineering(df_preprocessed)
    print(f"Features shape: {df_features.shape}")
    
    # 3. Handle Large Gaps
    print("Dropping remaining large missing data gaps from targets...")
    df_final = df_features.dropna(subset=["traffic_volume"])
    print(f"Final model-ready dataset shape: {df_final.shape}")
    
    # 4. Save Outputs
    print(f"Saving final dataset to {processed_path}...")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_final.to_csv(processed_path)
        
    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing and feature engineering pipeline.")
    parser.add_argument("--raw", type=str, default="data/raw/Metro_Interstate_Traffic_Volume.csv", help="Path to raw CSV file.")
    parser.add_argument("--processed", type=str, default="data/processed/clean_traffic_data.csv", help="Path to save processed CSV file.")
    args = parser.parse_args()
    
    run(args.raw, args.processed)
