from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import warnings


def train_model(X, y, test_size=0.2, random_state=0, n_estimators=300):
    # Decide whether to stratify based on class counts
    stratify = y
    try:
        vc = y.value_counts()
        if vc.min() < 2:
            warnings.warn('Some classes have fewer than 2 samples; skipping stratified split.')
            stratify = None
    except Exception:
        # If y is not a Series or lacks value_counts, skip stratify
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced'))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    # compute feature importances from underlying RF if available
    importances = None
    try:
        clf = pipe.named_steps['clf']
        feat_names = X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
        importances = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    except Exception:
        importances = None

    return pipe, acc, report, importances


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
