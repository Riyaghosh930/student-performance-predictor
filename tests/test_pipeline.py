import numpy as np  # type: ignore
from src.data_loader import DataLoader
from src.preprocessing import build_preprocessor
from src.feature_engineering import engineer_features

def test_data_loads():
    df = DataLoader().load()
    assert "final_grade" in df.columns
    assert len(df) > 0

def test_feature_engineering():
    df = engineer_features(DataLoader().load().head(50))
    assert "engagement_score" in df.columns
    assert df["engagement_score"].between(0, 1).all()

def test_preprocessor_transform():
    df = engineer_features(DataLoader().load().head(200))
    X = df.drop(columns=["student_id", "final_grade"])
    pre = build_preprocessor()
    Xp = pre.fit_transform(X)
    assert Xp.shape[0] == 200
    assert not np.isnan(Xp).any()