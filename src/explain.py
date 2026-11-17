"""SHAP explainability — global + per-student explanations."""
from pathlib import Path

try:
    import joblib  # type: ignore[import-not-found]
    import shap  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "Missing required runtime dependencies for explainability. "
        "Install the project dependencies (for example: pip install -r requirements.txt)."
    ) from exc

import numpy as np  # type: ignore[import-not-found]
from src.config import CONFIG
from src.data_loader import DataLoader
from src.feature_engineering import engineer_features

def explain(task: str = "regression", sample_size: int = 500):
    cfg = CONFIG
    out = Path(cfg.paths["reports"]); out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(Path(cfg.paths["models"]) / f"student_perf_{task}.joblib")
    pre, model = bundle["preprocessor"], bundle["model"]

    df = engineer_features(DataLoader().load())
    target = cfg.data["target"]
    X = df.drop(columns=[c for c in cfg.features.get("drop", []) + [target] if c in df.columns])
    Xs = pre.transform(X.sample(min(sample_size, len(X)), random_state=42))

    # Explain strongest base learner for speed; still reflects feature importances
    base_estimator = model.estimators_[0]  # xgboost
    explainer = shap.TreeExplainer(base_estimator)
    sv = explainer.shap_values(Xs)

    feat_names = (cfg.features["numeric"] + cfg.features["categorical"]
                  if hasattr(pre, "get_feature_names_out") is False else
                  [n.split("__")[-1] for n in pre.get_feature_names_out()])

    # Global summary
    plt.figure()
    shap.summary_plot(sv, Xs, feature_names=feat_names, show=False)
    plt.tight_layout(); plt.savefig(out / f"shap_summary_{task}.png", dpi=140, bbox_inches="tight")

    # Save raw values
    np.save(out / f"shap_values_{task}.npy", sv)
    print(f"SHAP explanations saved → {out}")

if __name__ == "__main__":
    explain("regression")