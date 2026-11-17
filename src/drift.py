"""Data drift detection via Evidently."""
from pathlib import Path
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from evidently.report import Report
from evidently.presets import DataDriftPreset
from src.config import CONFIG
from src.data_loader import DataLoader
from src.feature_engineering import engineer_features

def check_drift(baseline_path: str | None = None):
    cfg = CONFIG
    out = Path(cfg.paths["reports"]); out.mkdir(parents=True, exist_ok=True)
    current = engineer_features(DataLoader().load())
    if baseline_path and Path(baseline_path).exists():
        baseline = pd.read_csv(baseline_path)
    else:
        baseline = current.sample(frac=0.5, random_state=0)
        current = current.drop(baseline.index)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline, current_data=current)
    path = out / "drift_report.html"
    report.save_html(str(path))
    print(f"🛰️  Drift report → {path}")

if __name__ == "__main__":
    check_drift()