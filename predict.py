import argparse
from src.data import generate_synthetic_weather
from src.featurize import make_features
from src.model import load_model
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='model.joblib')
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # For demo, generate a few synthetic rows and predict
    df = generate_synthetic_weather(n_days=args.n_samples, seed=args.seed)
    X, y = make_features(df)
    model = load_model(args.model_path)
    preds = model.predict(X)
    out = df[['date', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind_speed']].copy()
    out['predicted'] = preds
    out['actual'] = y.values
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()

