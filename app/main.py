import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from typing import List
import traceback
from .schemas import AnalyzeSummary, PacketOut
from .utils import simulate_packet, parse_csv_bytes

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "rf_model.joblib"
MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))

app = FastAPI(title="Tenzorz / SentinelGuard IDS API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None
try:
    if Path(MODEL_PATH).exists():
        from .predictor import Predictor
        predictor = Predictor(MODEL_PATH)
        print("[*] Loaded model from", MODEL_PATH)
    else:
        print("[!] Model artefact not found at", MODEL_PATH, "- running in mock mode")
except Exception as e:
    print("[!] Failed loading predictor:", e)
    traceback.print_exc()
    predictor = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "model_path": str(MODEL_PATH)}

@app.get("/recent_packets", response_model=List[PacketOut])
def recent_packets(limit: int = 20):
    limit = max(1, min(200, limit))
    return [simulate_packet() for _ in range(limit)]

@app.post("/analyze", response_model=AnalyzeSummary)
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = parse_csv_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")

    total = len(df)
    if total == 0:
        raise HTTPException(status_code=400, detail="CSV file is empty or malformed.")

    if predictor is None:
        attack_count = int(total * 0.12)
        normal_count = total - attack_count
        summary = AnalyzeSummary(
            file=file.filename,
            total_rows=total,
            attack_count=attack_count,
            normal_count=normal_count,
            timestamp=datetime.utcnow()
        )
        return summary

    try:
        preds, _ = predictor.predict_df(df)
        attack_count = int((preds == 1).sum())
        normal_count = int((preds == 0).sum())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return AnalyzeSummary(
        file=file.filename,
        total_rows=total,
        attack_count=attack_count,
        normal_count=normal_count,
        timestamp=datetime.utcnow()
    )
