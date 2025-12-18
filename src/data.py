import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_weather(n_days=365, seed=0, out_csv=None):
    """Generate a richer synthetic daily weather dataset.

    Columns include: date, temp_max, temp_min, precipitation, humidity, wind_speed,
    pressure, visibility, cloud_cover, condition

    condition categories: sunny, cloudy, rainy, storm, fog, partly_cloudy, overcast
    """
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=n_days)
    dates = [start + timedelta(days=int(i)) for i in range(n_days)]

    # seasonal temperature pattern
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    base_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year / 365.0))
    temp_max = base_temp + rng.normal(0, 3, size=n_days) + 5
    temp_min = base_temp + rng.normal(0, 3, size=n_days) - 5

    # precipitation (mm), humidity (%), wind_speed (km/h)
    precipitation = np.clip(rng.gamma(1.2, 1.0, size=n_days) * (rng.random(n_days) < 0.35), 0, None)
    humidity = np.clip(45 + 40 * rng.random(n_days) + precipitation * 8, 0, 100)
    wind_speed = np.clip(rng.normal(8, 4, size=n_days), 0, None)

    # pressure (hPa) around typical sea-level pressure, slight seasonal noise
    pressure = np.round(1015 + rng.normal(0, 6, size=n_days) - 2 * np.sin(2 * np.pi * (day_of_year / 365.0)), 1)

    # visibility (km) inversely related to fog/precip/humidity
    visibility = np.clip(20 - (precipitation * 2) - (humidity - 50) * 0.05 + rng.normal(0, 2, size=n_days), 0.5, 20)

    # cloud cover percent
    cloud_cover = np.clip(20 + humidity * 0.5 + precipitation * 5 + rng.normal(0, 15, size=n_days), 0, 100)

    conditions = []
    for i in range(n_days):
        p = precipitation[i]
        h = humidity[i]
        w = wind_speed[i]
        vis = visibility[i]
        cc = cloud_cover[i]
        tmin = temp_min[i]

        # Rule-based label assignment with priority for storms/fog
        if p > 10 and w > 20:
            cond = 'storm'
        elif p > 2.5:
            cond = 'rainy'
        elif vis < 1.5 and h > 80:
            cond = 'fog'
        elif cc > 85:
            cond = 'overcast'
        elif cc > 60:
            cond = 'partly_cloudy'
        elif h > 75 and p > 0.5:
            cond = 'cloudy'
        else:
            cond = 'sunny'

        conditions.append(cond)

    # Ensure all classes are represented: force some random days into underrepresented classes
    all_classes = ['sunny', 'cloudy', 'rainy', 'storm', 'fog', 'partly_cloudy', 'overcast']
    present = set(conditions)
    missing = [c for c in all_classes if c not in present]
    for idx, cls in enumerate(missing):
        # overwrite a random day with this class
        j = int(rng.integers(0, n_days))
        conditions[j] = cls

    df = pd.DataFrame({
        'date': dates,
        'temp_max': np.round(temp_max, 1),
        'temp_min': np.round(temp_min, 1),
        'precipitation': np.round(precipitation, 2),
        'humidity': np.round(humidity, 1),
        'wind_speed': np.round(wind_speed, 1),
        'pressure': pressure,
        'visibility': np.round(visibility, 2),
        'cloud_cover': np.round(cloud_cover, 1),
        'condition': conditions,
    })

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df


def normalize_condition_labels(df, col='condition'):
    """Normalize string variants of conditions into canonical labels used by this project.

    Maps common variants (case insensitively) into: sunny, cloudy, rainy, storm, fog, partly_cloudy, overcast
    """
    if col not in df.columns:
        return df

    mapping = {
        'sunny': 'sunny', 'clear': 'sunny', 'clear skies': 'sunny',
        'cloudy': 'cloudy', 'clouds': 'cloudy',
        'partly cloudy': 'partly_cloudy', 'partly_cloudy': 'partly_cloudy', 'partly-cloudy': 'partly_cloudy',
        'overcast': 'overcast',
        'rain': 'rainy', 'rainy': 'rainy', 'showers': 'rainy',
        'storm': 'storm', 'thunderstorm': 'storm', 'thunderstorms': 'storm',
        'fog': 'fog', 'mist': 'fog', 'haze': 'fog'
    }

    def map_label(v):
        if pd.isna(v):
            return v
        key = str(v).strip().lower()
        return mapping.get(key, key.replace(' ', '_'))

    df[col] = df[col].apply(map_label)
    return df


def load_data(csv_path):
    """Load a CSV with weather data. Returns a pandas DataFrame.

    Expected columns: date, temp_max, temp_min, precipitation, humidity, wind_speed, condition
    This function will normalize the `condition` column if present.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = normalize_condition_labels(df, col='condition')
    return df
