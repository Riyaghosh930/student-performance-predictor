"""Robust data loading with schema validation and synthetic fallback."""
import logging
from pathlib import Path
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[reportMissingModuleSource]
from src.config import CONFIG

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, path: str | None = None):
        self.path = path or CONFIG.data["path"]

    def load(self) -> pd.DataFrame:
        if Path(self.path).exists():
            logger.info(f"Loading dataset from {self.path}")
            df = pd.read_csv(self.path)
        else:
            logger.warning("Dataset not found — generating synthetic dataset for demo.")
            df = self._synthesize()
        df = self._validate(df)
        return df

    def _synthesize(self, n: int = 5000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        study = rng.gamma(2.0, 3.0, n).clip(0, 40)
        attendance = rng.beta(8, 2, n) * 100
        prior_gpa = rng.normal(3.0, 0.5, n).clip(0, 4)
        sleep = rng.normal(7, 1.2, n).clip(3, 11)
        parental = rng.integers(0, 5, n)
        gender = rng.choice(["M", "F", "Other"], n, p=[0.48, 0.48, 0.04])
        school = rng.choice(["Public", "Private"], n, p=[0.7, 0.3])
        ses = rng.choice(["Low", "Mid", "High"], n, p=[0.3, 0.5, 0.2])
        internet = rng.choice(["Yes", "No"], n, p=[0.85, 0.15])

        noise = rng.normal(0, 5, n)
        final_grade = (
            2.2 * study
            + 0.45 * attendance
            + 8.0 * prior_gpa
            + 1.0 * sleep
            + 1.2 * parental
            + (internet == "Yes") * 3.0
            + noise
        )
        final_grade = np.clip(final_grade, 0, 100)

        return pd.DataFrame({
            "student_id": [f"S{i:05d}" for i in range(n)],
            "study_hours_per_week": study.round(2),
            "attendance_rate": attendance.round(2),
            "prior_gpa": prior_gpa.round(2),
            "sleep_hours": sleep.round(2),
            "parental_education_level": parental,
            "gender": gender,
            "school_type": school,
            "socioeconomic_status": ses,
            "internet_access": internet,
            "final_grade": final_grade.round(2),
        })

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        target = CONFIG.data["target"]
        assert target in df.columns, f"Missing target column '{target}'"
        n_before = len(df)
        df = df.dropna(subset=[target])
        if len(df) < n_before:
            logger.warning(f"Dropped {n_before - len(df)} rows with missing target.")
        return df