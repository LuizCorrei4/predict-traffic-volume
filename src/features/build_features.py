import pandas as pd
import numpy as np

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts temporal features (hour, day of week, month, year, weekend/rush hour flags)
    and computes cyclical encodings (sin/cos).
    """
    df_features = df.copy()
    
    # datetime index details
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    
    # Cyclical encodings
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24.0)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24.0)
    
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7.0)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7.0)
    
    # Weekend flag (Saturday=5, Sunday=6)
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    # Rush hour flag (7-9 AM, 3-6 PM on weekdays)
    is_weekday = df_features['day_of_week'] < 5
    is_morning_rush = (df_features['hour'] >= 7) & (df_features['hour'] <= 9)
    is_evening_rush = (df_features['hour'] >= 15) & (df_features['hour'] <= 18)
    df_features['is_rush_hour'] = (is_weekday & (is_morning_rush | is_evening_rush)).astype(int)
    
    return df_features

def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers robust indicators from weather variables and handles categorical labels."""
    df_features = df.copy()
    
    # Holiday flag
    if 'holiday' in df_features.columns:
        # holiday has NaN or string holiday name, or "None" string
        df_features['is_holiday'] = (
            df_features['holiday'].notna() & 
            (df_features['holiday'].astype(str) != 'None')
        ).astype(int)
        
    # Robust flags from weather_description (lowercase check to avoid case mismatches)
    desc = df_features['weather_description'].astype(str).str.lower()
    
    # is_raining
    rain_val = df_features['rain_1h'] if 'rain_1h' in df_features.columns else 0
    df_features['is_raining'] = (
        (rain_val > 0) | 
        desc.str.contains('rain') | 
        desc.str.contains('drizzle')
    ).astype(int)
    
    # is_snowing (resolves snow sensor silent failures)
    snow_val = df_features['snow_1h'] if 'snow_1h' in df_features.columns else 0
    df_features['is_snowing'] = (
        (snow_val > 0) | 
        desc.str.contains('snow') | 
        desc.str.contains('sleet')
    ).astype(int)
    
    # is_foggy_misty
    df_features['is_foggy_misty'] = (
        desc.str.contains('fog') | 
        desc.str.contains('mist') | 
        desc.str.contains('haze')
    ).astype(int)
    
    # Log transform rain_1h due to right skewness
    if 'rain_1h' in df_features.columns:
        df_features['rain_1h_log'] = np.log1p(df_features['rain_1h'].fillna(0))
        
    return df_features

def pipeline_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full feature engineering pipeline (temporal and weather features)."""
    df_engineered = extract_temporal_features(df)
    df_engineered = engineer_weather_features(df_engineered)
    
    return df_engineered
