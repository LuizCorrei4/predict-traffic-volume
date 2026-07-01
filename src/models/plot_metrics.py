"""
Script to generate visualization figures for the final report.
It plots metric comparisons (from JSON logs) and Time Series actual vs predicted
(Full Year, One Month, One Week, and 3 Days) for the best models.
"""

import os
import json
import glob
import joblib
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic style for the plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

def load_latest_metrics(logs_dir="reports/logs"):
    """Reads the latest JSON log for each algorithm."""
    all_files = glob.glob(os.path.join(logs_dir, "*.json"))
    
    # Group files by algorithm
    algo_files = {}
    for f in all_files:
        algo = os.path.basename(f).split("_")[0]
        if algo not in algo_files:
            algo_files[algo] = []
        algo_files[algo].append(f)
        
    metrics = []
    for algo, files in algo_files.items():
        # Get the most recently modified file for the algorithm
        latest_file = max(files, key=os.path.getmtime)
        with open(latest_file, 'r') as jf:
            data = json.load(jf)
            metrics.append({
                "Algorithm": data["algorithm"].upper(),
                "RMSE": data["metrics"]["RMSE"],
                "MAPE": data["metrics"]["MAPE_percentage"],
                "R2": data["metrics"]["R2"]
            })
            
    return pd.DataFrame(metrics)

def plot_metrics_comparison(df_metrics, output_dir="reports/figures"):
    """Generates bar charts comparing algorithm performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by MAPE ascending
    df_metrics = df_metrics.sort_values(by="MAPE")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. MAPE
    sns.barplot(x="Algorithm", y="MAPE", data=df_metrics, ax=axes[0], hue="Algorithm", legend=False, palette="Blues_d")
    axes[0].set_title("Mean Absolute Percentage Error (MAPE)")
    axes[0].set_ylabel("Erro (%)")
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. RMSE
    sns.barplot(x="Algorithm", y="RMSE", data=df_metrics.sort_values(by="RMSE"), ax=axes[1], hue="Algorithm", legend=False, palette="Reds_d")
    axes[1].set_title("Root Mean Squared Error (RMSE)")
    axes[1].set_ylabel("Carros/Hora")
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. R2
    sns.barplot(x="Algorithm", y="R2", data=df_metrics.sort_values(by="R2", ascending=False), ax=axes[2], hue="Algorithm", legend=False, palette="Greens_d")
    axes[2].set_title("R² Score")
    axes[2].set_ylabel("Variância Explicada")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Metrics comparison plot saved to {filepath}")

def plot_time_series_predictions(best_model_name, mode="test", output_dir="reports/figures"):
    """
    Loads the validation or test data and the serialized model, then plots
    Actual vs Predicted for the entire set, 1 month, 1 week, and 3 days.
    """
    model_path = f"models/{best_model_name.lower()}_best.pkl"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping timeseries plots.")
        return
        
    print(f"Loading {best_model_name} for Time Series Plotting on {mode.upper()} set...")
    model = joblib.load(model_path)
    
    # Load dataset based on mode
    if mode == "test":
        X = pd.read_csv("data/processed/X_test_scaled.csv", index_col="date_time", parse_dates=True)
        y = pd.read_csv("data/processed/y_test.csv", index_col="date_time", parse_dates=True)
    else:
        X = pd.read_csv("data/processed/X_val_scaled.csv", index_col="date_time", parse_dates=True)
        y = pd.read_csv("data/processed/y_val.csv", index_col="date_time", parse_dates=True)
    
    # Ensure chronological order
    X = X.sort_index()
    y = y.sort_index()
    
    preds = model.predict(X)
    
    df_plot = pd.DataFrame({
        "Actual": y["traffic_volume"],
        "Predicted": preds
    }, index=y.index)
    
    # Plot 1: Full Range (To show overall trend)
    plt.figure(figsize=(15, 5))
    df_plot_resampled = df_plot.resample('1D').mean()
    plt.plot(df_plot_resampled.index, df_plot_resampled["Actual"], label="Real (Média Diária)", alpha=0.7)
    plt.plot(df_plot_resampled.index, df_plot_resampled["Predicted"], label="Previsto (Média Diária)", alpha=0.7)
    plt.title(f"{best_model_name} - Previsão de Tráfego (Visão Geral - {mode.upper()})")
    plt.xlabel("Data")
    plt.ylabel("Volume de Tráfego")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{best_model_name}_ts_full.png"), dpi=300)
    plt.close()
    
    # Plot 2: One Month Slice
    mid_idx = len(df_plot) // 2
    mid_date = df_plot.index[mid_idx]
    start_month = f"{mid_date.year}-{mid_date.month:02d}-01"
    end_month = pd.to_datetime(start_month) + pd.DateOffset(months=1)
    
    df_month = df_plot[start_month:end_month.strftime('%Y-%m-%d')]
    if not df_month.empty:
        plt.figure(figsize=(15, 5))
        plt.plot(df_month.index, df_month["Actual"], label="Real", alpha=0.8)
        plt.plot(df_month.index, df_month["Predicted"], label="Previsto", alpha=0.8)
        plt.title(f"{best_model_name} - Previsão de Tráfego (Recorte de 1 Mês)")
        plt.xlabel("Data")
        plt.ylabel("Volume de Tráfego")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{best_model_name}_ts_month.png"), dpi=300)
        plt.close()
        
        # Plot 3: One Week Slice
        start_week = df_month.index[0]
        end_week = start_week + pd.DateOffset(days=7)
        df_week = df_month[start_week.strftime('%Y-%m-%d'):end_week.strftime('%Y-%m-%d')]
        
        plt.figure(figsize=(15, 5))
        plt.plot(df_week.index, df_week["Actual"], label="Real", linewidth=2)
        plt.plot(df_week.index, df_week["Predicted"], label="Previsto", linewidth=2, linestyle='--')
        plt.title(f"{best_model_name} - Previsão de Tráfego (Recorte de 1 Semana)")
        plt.xlabel("Data")
        plt.ylabel("Volume de Tráfego")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{best_model_name}_ts_week.png"), dpi=300)
        plt.close()
        
        # Plot 4: 3 Days Slice (Micro-trends)
        start_3d = df_week.index[0]
        end_3d = start_3d + pd.DateOffset(days=3)
        df_3d = df_week[start_3d.strftime('%Y-%m-%d'):end_3d.strftime('%Y-%m-%d')]
        
        plt.figure(figsize=(15, 5))
        plt.plot(df_3d.index, df_3d["Actual"], label="Real", linewidth=2, marker='o', markersize=4)
        plt.plot(df_3d.index, df_3d["Predicted"], label="Previsto", linewidth=2, linestyle='--', marker='x', markersize=4)
        plt.title(f"{best_model_name} - Previsão de Tráfego (Recorte de 3 Dias - Microtendências)")
        plt.xlabel("Data")
        plt.ylabel("Volume de Tráfego")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{best_model_name}_ts_3days.png"), dpi=300)
        plt.close()
        
    print(f"Time Series plots saved for {best_model_name}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", choices=["validation", "test"], 
                        help="Dataset mode to evaluate (validation or test)")
    args = parser.parse_args()

    print("Generating Visualization Figures...")
    df_metrics = load_latest_metrics()
    
    if not df_metrics.empty:
        print("Metrics found:")
        print(df_metrics.sort_values(by="MAPE"))
        plot_metrics_comparison(df_metrics)
        
        # Identify the best algorithm (Lowest MAPE)
        best_algo = df_metrics.sort_values(by="MAPE").iloc[0]["Algorithm"]
        print(f"\nBest Algorithm Identified: {best_algo}")
        
        # Plot time series for the best algorithm
        plot_time_series_predictions(best_algo, mode=args.mode)
        
    else:
        print("No metrics logs found in reports/logs/")

if __name__ == "__main__":
    main()
