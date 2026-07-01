# Visualization Figures

This directory contains the final generated visual reports for the **predict-traffic-volume** project. 

The plots were automatically extracted by the `src/models/plot_metrics.py` script after executing the Machine Learning pipeline in both `validation` and `test` modes.

## 1. Model Comparison (`metrics_comparison.png`)
This image contains a side-by-side bar chart comparison across all the implemented algorithms (Ridge, Random Forest, XGBoost, LightGBM, CatBoost, and MLP). 
It compares the evaluation metrics (RMSE, MAPE, and $R^2$) to easily identify which algorithm handled the stochastic nature of traffic data best during the cross-validated Bayesian tuning phase.

## 2. Time Series: Actual vs Predicted (Random Forest)
The algorithm that emerged victorious from the Deep Hyperparameter Tuning (150 trials using Optuna) was the **Random Forest Regressor**. 

The following images were plotted against the isolated **Test Set** (data the model was never trained on) to prove its real-world generalization power. The model achieved a stellar $R^2$ of **0.9502** and a MAPE of **10.7%**.

To properly showcase how well the algorithm captures cyclical human behavior over time, the predictions are broken down into four distinct temporal resolutions:

- **`RANDOM_FOREST_ts_full.png`**: The macro-view. Displays the daily moving average of traffic over the entire duration of the Test set (roughly spanning a year), showing overall seasonal trends.
- **`RANDOM_FOREST_ts_month.png`**: A 30-day slice. It clearly shows the weekend dips and weekday peaks across a month.
- **`RANDOM_FOREST_ts_week.png`**: A 7-day slice. Ideal for observing the distinct morning and afternoon rush-hour spikes on workdays versus the flatter distributions on weekends.
- **`RANDOM_FOREST_ts_3days.png`**: The micro-trend view. A zoomed-in 72-hour window with node markers to prove the model's hour-by-hour responsiveness to night/day cycles and changing weather patterns.
