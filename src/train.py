"""Stacking ensemble with Optuna hyperparameter search."""
import logging, joblib
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, StackingRegressor, GradientBoostingRegressor,
    RandomForestClassifier, StackingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from src.config import CONFIG
from src.data_loader import DataLoader
from src.preprocessing import build_preprocessor
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_base_models(task: str, params: dict):
    if task == "regression":
        return [
            ("xgb", XGBRegressor(**params.get("xgb", {}), n_jobs=-1, verbosity=0)),
            ("lgbm", LGBMRegressor(**params.get("lgbm", {}), n_jobs=-1, verbose=-1)),
            ("cat", CatBoostRegressor(**params.get("cat", {}), verbose=0)),
            ("rf", RandomForestRegressor(**params.get("rf", {}), n_jobs=-1)),
        ]
    else:
        return [
            ("xgb", XGBClassifier(**params.get("xgb", {}), n_jobs=-1, verbosity=0, eval_metric="logloss")),
            ("lgbm", LGBMClassifier(**params.get("lgbm", {}), n_jobs=-1, verbose=-1)),
            ("cat", CatBoostClassifier(**params.get("cat", {}), verbose=0)),
            ("rf", RandomForestClassifier(**params.get("rf", {}), n_jobs=-1)),
        ]


def _optuna_search(X, y, task: str, n_trials: int, cv: int, seed: int) -> dict:
    """Tune the strongest single model (LightGBM) then share sensible defaults."""
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        )
        if task == "regression":
            model = LGBMRegressor(**params, n_jobs=-1, verbose=-1)
            score = -cross_val_score(model, X, y, cv=cv,
                                     scoring="neg_root_mean_squared_error").mean()
        else:
            model = LGBMClassifier(**params, n_jobs=-1, verbose=-1)
            score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
        return score

    study = optuna.create_study(direction="minimize" if task == "regression" else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best Optuna params: {study.best_params}  |  score={study.best_value:.4f}")
    return study.best_params


def train(task: str = "regression"):
    cfg = CONFIG
    cfg.ensure_dirs()
    df = DataLoader().load()
    df = engineer_features(df)

    target = cfg.data["target"]
    drop_cols = cfg.features.get("drop", []) + [target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]

    if task == "classification":
        threshold = cfg.data.get("classification_threshold", 60)
        y = (y >= threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data["test_size"],
        random_state=cfg.data["random_state"],
        stratify=y if task == "classification" else None,
    )

    pre = build_preprocessor()
    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    cv = cfg.model["cv_folds"]
    best_lgbm = _optuna_search(X_train_p, y_train, task,
                               cfg.model["optuna_trials"], cv,
                               cfg.data["random_state"])

    # Sensible defaults derived from best LGBM params
    shared = dict(
        n_estimators=best_lgbm["n_estimators"],
        learning_rate=best_lgbm["learning_rate"],
        max_depth=best_lgbm["max_depth"],
    )
    params = {
        "xgb": dict(n_estimators=shared["n_estimators"], learning_rate=shared["learning_rate"],
                    max_depth=shared["max_depth"], subsample=0.9, colsample_bytree=0.9),
        "lgbm": best_lgbm,
        "cat": dict(iterations=shared["n_estimators"], learning_rate=shared["learning_rate"],
                    depth=shared["max_depth"]),
        "rf": dict(n_estimators=400, max_depth=None, min_samples_leaf=2),
    }

    base = _make_base_models(task, params)
    meta = Ridge() if task == "regression" else LogisticRegression(max_iter=1000)

    stacker_cls = StackingRegressor if task == "regression" else StackingClassifier
    stacker = stacker_cls(
        estimators=base,
        final_estimator=meta,
        cv=KFold(n_splits=cv, shuffle=True, random_state=cfg.data["random_state"]),
        n_jobs=-1,
        passthrough=True,
    )
    stacker.fit(X_train_p, y_train)

    # Evaluate
    y_pred = stacker.predict(X_test_p)
    if task == "regression":
        metrics = dict(
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            r2=float(r2_score(y_test, y_pred)),
        )
    else:
        y_proba = stacker.predict_proba(X_test_p)[:, 1]
        metrics = dict(
            accuracy=float(accuracy_score(y_test, y_pred)),
            roc_auc=float(roc_auc_score(y_test, y_proba)),
        )
    logger.info(f"[{task}] Test metrics: {metrics}")

    # Persist
    out = Path(cfg.paths["models"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": pre, "model": stacker, "metrics": metrics,
                 "task": task, "target": target},
                out / f"student_perf_{task}.joblib")
    logger.info(f"Saved model → {out / f'student_perf_{task}.joblib'}")
    return metrics


if __name__ == "__main__":
    train("regression")
    train("classification")