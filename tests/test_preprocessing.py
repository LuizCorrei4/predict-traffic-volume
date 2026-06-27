import pandas as pd
import numpy as np
from src.data.preprocessing import resolve_duplicates, handle_outliers

def test_resolve_duplicates():
    # Create sample DataFrame with duplicate timestamps
    dates = pd.to_datetime(['2023-01-01 01:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00'])
    data = {
        'temp': [280.0, 290.0, 270.0],
        'weather_main': ['Clouds', 'Clear', 'Rain'],
        'traffic_volume': [1000, 2000, 1500]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date_time'
    
    df_agg = resolve_duplicates(df)
    
    assert len(df_agg) == 2
    # Mean for numeric
    assert df_agg.loc['2023-01-01 01:00:00', 'temp'] == 285.0
    assert df_agg.loc['2023-01-01 01:00:00', 'traffic_volume'] == 1500
    # First for categorical
    assert df_agg.loc['2023-01-01 01:00:00', 'weather_main'] == 'Clouds'

def test_handle_outliers():
    dates = pd.to_datetime(['2023-01-01 01:00:00', '2023-01-01 02:00:00'])
    data = {
        'temp': [0.0, 290.0],  # 0.0 is an outlier (<100K)
        'rain_1h': [9831.3, 0.0],  # 9831.3 is an outlier (>500mm)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date_time'
    
    df_clean = handle_outliers(df)
    
    assert np.isnan(df_clean.loc['2023-01-01 01:00:00', 'temp'])
    assert np.isnan(df_clean.loc['2023-01-01 01:00:00', 'rain_1h'])
    assert df_clean.loc['2023-01-01 02:00:00', 'temp'] == 290.0
    assert df_clean.loc['2023-01-01 02:00:00', 'rain_1h'] == 0.0
