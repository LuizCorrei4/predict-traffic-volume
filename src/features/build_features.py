import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

def fit_encoders(df: pd.DataFrame):
    """Fits and returns standardizing scalers and one-hot encoders to avoid data leakage."""
    scalers = {}
    
    # Fit StandardScaler for temperature
    if 'temp' in df.columns:
        temp_scaler = StandardScaler()
        temp_scaler.fit(df[['temp']].fillna(df['temp'].mean()))
        scalers['temp_scaler'] = temp_scaler
        
    # Fit OneHotEncoder for weather_main
    if 'weather_main' in df.columns:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # We fillna with 'Clear' as it is a common baseline
        ohe.fit(df[['weather_main']].fillna('Clear'))
        scalers['weather_main_ohe'] = ohe
        
    return scalers

def transform_features(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Applies the pre-fitted scalers and encoders to the dataframe."""
    df_features = df.copy()
    
    # Scale temp
    if 'temp' in df_features.columns and 'temp_scaler' in scalers:
        mean_val = scalers['temp_scaler'].mean_[0]
        df_features['temp_scaled'] = scalers['temp_scaler'].transform(
            df_features[['temp']].fillna(mean_val)
        )
        
    # One-hot encode weather_main
    if 'weather_main' in df_features.columns and 'weather_main_ohe' in scalers:
        ohe = scalers['weather_main_ohe']
        ohe_cols = ohe.get_feature_names_out(['weather_main'])
        
        encoded_arr = ohe.transform(df_features[['weather_main']].fillna('Clear'))
        encoded_df = pd.DataFrame(encoded_arr, columns=ohe_cols, index=df_features.index)
        
        df_features = pd.concat([df_features, encoded_df], axis=1)
        
    return df_features

def pipeline_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full feature engineering pipeline (temporal and weather features).
    Scalers and encoders should be applied later, after train-test split.
    """
    df_engineered = extract_temporal_features(df)
    df_engineered = engineer_weather_features(df_engineered)
    
    return df_engineered
