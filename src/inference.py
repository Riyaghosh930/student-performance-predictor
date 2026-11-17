"""Reusable inference service."""
from pathlib import Path
import joblib 
import pandas as pd
from src.config import CONFIG
from src.feature_engineering import engineer_features

class StudentPerformancePredictor:
    def __init__(self, task: str = "regression"):
        cfg = CONFIG
        bundle = joblib.load(Path(cfg.paths["models"]) / f"student_perf_{task}.joblib")
        self.pre = bundle["preprocessor"]
        self.model = bundle["model"]
        self.task = task

    def predict(self, records: list[dict]) -> list[dict]:
        df = engineer_features(pd.DataFrame(records))
        Xp = self.pre.transform(df)
        if self.task == "regression":
            preds = self.model.predict(Xp).clip(0, 100)
            return [{"predicted_grade": float(p)} for p in preds]
        else:
            proba = self.model.predict_proba(Xp)[:, 1]
            return [{"pass_probability": float(p), "prediction": int(p >= 0.5)}
                    for p in proba]