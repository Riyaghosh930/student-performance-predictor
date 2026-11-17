"""Deeper evaluation: ROC, PR curve, calibration, confusion matrix, residuals."""
import json
from pathlib import Path
import importlib.util

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (  # pyright: ignore[reportMissingModuleSource]
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingModuleSource]
from src.config import CONFIG
from src.data_loader import DataLoader
from src.feature_engineering import engineer_features

if importlib.util.find_spec("joblib") is not None:
    import joblib  # pyright: ignore[reportMissingImports]
else:
    joblib = None


def evaluate(task: str = "classification"):
    cfg = CONFIG
    cfg.ensure_dirs()
    if joblib is None:
        raise ImportError("joblib is required to load the trained model bundle.")
    bundle = joblib.load(Path(cfg.paths["models"]) / f"student_perf_{task}.joblib")
    pre, model = bundle["preprocessor"], bundle["model"]

    df = engineer_features(DataLoader().load())
    target = cfg.data["target"]
    X = df.drop(columns=[c for c in cfg.features.get("drop", []) + [target] if c in df.columns])
    y = df[target]
    if task == "classification":
        y = (y >= cfg.data["classification_threshold"]).astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.data["test_size"],
        random_state=cfg.data["random_state"],
        stratify=y if task == "classification" else None,
    )
    Xp = pre.transform(X_test)
    out = Path(cfg.paths["reports"]); out.mkdir(parents=True, exist_ok=True)

    if task == "classification":
        yp = model.predict(Xp)
        proba = model.predict_proba(Xp)[:, 1]
        cm = confusion_matrix(y_test, yp)
        report = classification_report(y_test, yp, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, proba); roc_auc = auc(fpr, tpr)
        pr, rc, _ = precision_recall_curve(y_test, proba)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ConfusionMatrixDisplay(cm).plot(ax=axes[0], colorbar=False)
        axes[0].set_title("Confusion Matrix")
        axes[1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); axes[1].plot([0,1],[0,1],'--')
        axes[1].set_title("ROC"); axes[1].legend()
        axes[2].plot(rc, pr); axes[2].set_title("Precision-Recall")
        plt.tight_layout(); plt.savefig(out / "classification_report.png", dpi=140)

        json.dump({"classification_report": report, "roc_auc": roc_auc,
                   "confusion_matrix": cm.tolist()},
                  open(out / "classification_metrics.json", "w"), indent=2)
    else:
        yp = model.predict(Xp)
        resid = y_test - yp
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(yp, resid, alpha=0.4); axes[0].axhline(0, color="r")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residuals")
        axes[1].hist(resid, bins=40); axes[1].set_title("Residual Distribution")
        plt.tight_layout(); plt.savefig(out / "regression_report.png", dpi=140)
        json.dump({"rmse": float(np.sqrt(((y_test - yp)**2).mean())),
                   "mae": float(np.abs(y_test - yp).mean())},
                  open(out / "regression_metrics.json", "w"), indent=2)

    print(f"📊 Reports written to {out}")

if __name__ == "__main__":
    evaluate("classification"); evaluate("regression")