import pandas as pd
import numpy as np


def add_time_features(df, date_col='date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek
    # cyclic encoding for day_of_year and month
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    df['sin_month'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    return df


def make_features(df):
    """Return feature matrix X and labels y.

    - If the input DataFrame contains a 'condition' column, return (X, y).
    - Otherwise return (X, None) so prediction code can proceed for single rows.
    """
    df = add_time_features(df)
    features = [
        'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind_speed',
        'pressure', 'visibility', 'cloud_cover',
        'sin_day', 'cos_day', 'sin_month', 'cos_month', 'day_of_week'
    ]
    X = df[features].fillna(0)
    y = df['condition'] if 'condition' in df.columns else None
    return X, y
