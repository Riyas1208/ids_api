"""
train_model.py

Usage:
    python train_model.py --data data/nsl_kdd_preprocessed.csv --out models/rf_model.joblib

This script:
 - Loads a CSV expected to have a 'label' column (label values: 'normal' or attack-class or 0/1)
 - Selects numeric features (you can change selection for NSL-KDD)
 - Scales features (StandardScaler)
 - Trains a RandomForestClassifier
 - Saves an artefact (dict) with 'model', 'scaler', and 'feature_cols' to joblib
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def load_and_preprocess(path):
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column indicating normal/attack")

    # Convert string labels to binary: 0 = normal, 1 = attack
    if df['label'].dtype == object:
        df['label'] = df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)

    # For NSL-KDD, you likely have categorical columns; this script selects numeric features only.
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric:
        numeric.remove('label')
    if len(numeric) == 0:
        raise ValueError("No numeric features found. Preprocess dataset to include numeric features (or use one-hot encoding).")

    X = df[numeric]
    y = df['label'].astype(int)

    return X, y, numeric

def main(args):
    data_path = args.data
    out_path = args.out

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training CSV not found: {data_path}")

    print("[*] Loading and preprocessing")
    X, y, feature_cols = load_and_preprocess(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("[*] Training RandomForest")
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"[*] Test accuracy: {acc:.4f}")
    print("[*] Classification report:")
    print(classification_report(y_test, preds))

    artefact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artefact, out_path)
    print("[*] Saved artefact to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training CSV (must include label column)")
    parser.add_argument("--out", default="models/rf_model.joblib", help="Output path for joblib artefact")
    args = parser.parse_args()
    main(args)
