import pandas as pd

def make_features(df: pd.DataFrame):
    """
    Converts weather dataframe into numerical feature set
    for machine learning prediction.
    """
    df = df.copy()

    # Convert date into numeric values
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Feature columns (for ML model)
    feature_cols = [
        'temp_max', 'temp_min', 'humidity', 'precipitation',
        'pressure', 'visibility', 'wind_speed', 'cloud_cover',
        'day', 'month', 'year'
    ]

    X = df[feature_cols]

    # If weather_type exists (during training)
    y = df['weather_type'] if 'weather_type' in df.columns else None

    return X, y
