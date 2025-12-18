import pandas as pd
from pathlib import Path
import joblib
from src.featurize import make_features

MODEL_PATH = Path('model.joblib')
SAMPLE_CSV = Path('data/sample_weather.csv')

assert MODEL_PATH.exists(), 'model.joblib not found'
assert SAMPLE_CSV.exists(), 'sample_weather.csv not found'

print('Loading sample data...')
df = pd.read_csv(SAMPLE_CSV, parse_dates=['date'])
print('Total rows:', len(df))
print('Class distribution:')
print(df['condition'].value_counts())

print('\nLoading model...')
model = joblib.load(MODEL_PATH)

print('Model type:', type(model))
try:
    clf = model.named_steps['clf']
    print('Underlying classifier:', type(clf))
    if hasattr(clf, 'classes_'):
        print('Model classes:', clf.classes_)
    if hasattr(clf, 'feature_importances_'):
        importances = sorted(zip(df.columns.tolist(), [0]*len(df.columns)), key=lambda x: x[1])
        # better to print named features from featurizer
        X_sample, _ = make_features(df.head(1))
        feat_names = X_sample.columns.tolist()
        importances = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
        print('\nTop feature importances:')
        for name, imp in importances[:10]:
            print(f'  {name}: {imp:.4f}')
except Exception as e:
    print('Could not inspect classifier internals:', e)

# Test specific cases
cases = [
    {'name': 'Hot dry', 'temp_max': 35, 'temp_min': 25, 'precipitation': 0.0, 'humidity': 30, 'wind_speed': 3},
    {'name': 'Cold snowy', 'temp_max': -1, 'temp_min': -5, 'precipitation': 5.0, 'humidity': 85, 'wind_speed': 5},
    {'name': 'Rainy', 'temp_max': 18, 'temp_min': 12, 'precipitation': 3.0, 'humidity': 90, 'wind_speed': 6},
    {'name': 'Cloudy high humidity', 'temp_max': 22, 'temp_min': 16, 'precipitation': 0.0, 'humidity': 80, 'wind_speed': 4},
    {'name': 'Edge low precip', 'temp_max': 10, 'temp_min': 2, 'precipitation': 0.6, 'humidity': 60, 'wind_speed': 2},
]

print('\nPredictions for test cases:')
for c in cases:
    pressure = 1015.0
    visibility = max(0.5, 20 - (c['precipitation'] * 2) - (c['humidity'] - 50) * 0.05)
    cloud_cover = min(100.0, max(0.0, 20 + c['humidity'] * 0.5 + c['precipitation'] * 5))
    df_row = pd.DataFrame([{
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
    X, _ = make_features(df_row)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
    print(c['name'], '=>', pred, 'proba=', dict(zip(model.named_steps['clf'].classes_, proba)) if proba is not None else None)

