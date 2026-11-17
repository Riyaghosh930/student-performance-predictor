"""Leak-free preprocessing pipeline using ColumnTransformer."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from src.config import CONFIG

def build_preprocessor() -> ColumnTransformer:
    num = CONFIG.features["numeric"]
    cat = CONFIG.features["categorical"]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # TargetEncoder (sklearn >=1.3) handles high-cardinality categoricals smartly
    try:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    except TypeError:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num),
            ("cat", cat_pipe, cat),
        ],
        remainder="drop",
    )