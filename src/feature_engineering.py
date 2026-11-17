"""Domain-driven feature engineering."""
import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Composite "engagement" score
    df["engagement_score"] = (
        df["study_hours_per_week"].clip(0, 40) / 40 * 0.5
        + df["attendance_rate"].clip(0, 100) / 100 * 0.3
        + df["sleep_hours"].clip(3, 10) / 10 * 0.2
    )
    # Study-efficiency proxy
    df["study_efficiency"] = df["study_hours_per_week"] / (
        df["sleep_hours"].clip(lower=1) + 1
    )
    # Risk flag
    df["at_risk_flag"] = (
        (df["attendance_rate"] < 70) | (df["prior_gpa"] < 2.0)
    ).astype(int)
    # Interaction
    df["gpa_x_attendance"] = df["prior_gpa"] * df["attendance_rate"] / 100
    return df