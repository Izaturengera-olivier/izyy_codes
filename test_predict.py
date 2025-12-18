import pandas as pd
from pathlib import Path
import joblib
from src.featurize import make_features

MODEL_PATH = Path('model.joblib')
assert MODEL_PATH.exists(), 'model.joblib not found'
model = joblib.load(MODEL_PATH)

cases = [
    {'name': 'Hot dry', 'temp_max': 35, 'temp_min': 25, 'precipitation': 0.0, 'humidity': 30, 'wind_speed': 3},
    {'name': 'Cold snowy', 'temp_max': -1, 'temp_min': -5, 'precipitation': 5.0, 'humidity': 85, 'wind_speed': 5},
    {'name': 'Rainy', 'temp_max': 18, 'temp_min': 12, 'precipitation': 3.0, 'humidity': 90, 'wind_speed': 6},
    {'name': 'Cloudy high humidity', 'temp_max': 22, 'temp_min': 16, 'precipitation': 0.0, 'humidity': 80, 'wind_speed': 4},
    {'name': 'Edge low precip', 'temp_max': 10, 'temp_min': 2, 'precipitation': 0.6, 'humidity': 60, 'wind_speed': 2},
]

for c in cases:
    pressure = 1015.0
    visibility = max(0.5, 20 - (c['precipitation'] * 2) - (c['humidity'] - 50) * 0.05)
    cloud_cover = min(100.0, max(0.0, 20 + c['humidity'] * 0.5 + c['precipitation'] * 5))
    df = pd.DataFrame([{
        'date': pd.Timestamp('2025-12-11'),
        'temp_max': c['temp_max'],
        'temp_min': c['temp_min'],
        'precipitation': c['precipitation'],
        'humidity': c['humidity'],
        'wind_speed': c['wind_speed'],
        'pressure': pressure,
        'visibility': visibility,
        'cloud_cover': cloud_cover,
    }])
    X, y = make_features(df)
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
    print(c['name'], '=>', pred, 'proba=', proba)
