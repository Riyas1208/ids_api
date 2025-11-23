import joblib
import numpy as np
import pandas as pd
from typing import Tuple
import os

class Predictor:
    def __init__(self, artefact_path: str):
        if not os.path.exists(artefact_path):
            raise FileNotFoundError(f"Model artefact not found: {artefact_path}")
        artefact = joblib.load(artefact_path)
        self.model = artefact.get('model')
        self.scaler = artefact.get('scaler')  # may be None
        self.feature_cols = artefact.get('feature_cols')
        if self.model is None or self.feature_cols is None:
            raise ValueError("Model artefact must contain 'model' and 'feature_cols' keys.")

    def predict_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in input CSV: {missing}")

        X = df[self.feature_cols].astype(float).values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        preds = self.model.predict(X).astype(int)
        probs = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else np.zeros(len(preds))
        return preds, probs
