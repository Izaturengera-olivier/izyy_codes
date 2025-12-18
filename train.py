import argparse
from src.data import generate_synthetic_weather, load_data
from src.featurize import make_features
from src.model import train_model, save_model
import os
import pandas as pd
import numpy as np


def balance_classes(X, y, random_state=0):
    df = X.copy()
    df['__target'] = y.values
    # find max class count and upsample others
    counts = df['__target'].value_counts()
    max_n = counts.max()
    parts = []
    rng = np.random.RandomState(random_state)
    for cls, cnt in counts.items():
        sub = df[df['__target'] == cls]
        if cnt < max_n:
            rep = sub.sample(max_n, replace=True, random_state=random_state)
            parts.append(rep)
        else:
            parts.append(sub)
    balanced = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_bal = balanced['__target']
    X_bal = balanced.drop(columns=['__target'])
    return X_bal, y_bal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-sample', action='store_true')
    parser.add_argument('--n-days', type=int, default=365)
    parser.add_argument('--sample-path', type=str, default='data/sample_weather.csv')
    parser.add_argument('--model-path', type=str, default='model.joblib')
    parser.add_argument('--balance', action='store_true', help='Balance classes by upsampling minority classes')
    args = parser.parse_args()

    if args.generate_sample:
        print('Generating synthetic data to', args.sample_path)
        df = generate_synthetic_weather(n_days=args.n_days, out_csv=args.sample_path)
    else:
        df = load_data(args.sample_path)

    X, y = make_features(df)
    print('Training model on', len(X), 'rows')

    if args.balance:
        print('Balancing classes by upsampling...')
        X, y = balance_classes(X, y)
        print('Balanced dataset size:', len(X))

    model, acc, report, importances = train_model(X, y)
    print('Accuracy:', acc)
    print(report)
    if importances:
        print('Top features:')
        for name, imp in importances[:10]:
            print(f'  {name}: {imp:.4f}')
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    save_model(model, args.model_path)
    print('Saved model to', args.model_path)


if __name__ == '__main__':
    main()
