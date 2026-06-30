"""
Configuration file defining the final feature set for modeling.
Based strictly on the conclusions from:
notebooks/02-eda_after_preprocess_insights.md
"""

# The target variable we want to predict
TARGET_FEATURE = "traffic_volume"

# Columns to drop because they are redundant, cause multicollinearity, 
# or are of low quality/faulty (e.g. snow_1h, raw text, raw dates).
FEATURES_TO_DROP = [
    "holiday",              # Replaced by is_holiday
    "weather_main",         # Replaced by boolean flags
    "weather_description",  # Replaced by boolean flags
    "hour",                 # Replaced by hour_sin/cos
    "day_of_week",          # Replaced by day_sin/cos
    "month",                # Replaced by cyclical features
    "year",                 # Dropped to avoid non-stationary trending confusion
    "rain_1h",              # Replaced by rain_1h_log
    "snow_1h"               # Dropped due to silent sensor failures (using is_snowing instead)
]

# Continuous numerical features that require scaling (e.g. StandardScaler)
NUMERICAL_FEATURES_TO_SCALE = [
    "temp",
    "clouds_all",
    "rain_1h_log"
]

# Note: Cyclical sine/cosine features and binary flags (is_weekend, etc.) 
# do not require scaling as they are already bounded/categorical in nature.
