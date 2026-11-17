"""Interactive dashboard for live prediction + insights."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# ... the rest of your existing imports (streamlit, plotly, src.inference, etc.)
import streamlit as st, pandas as pd, plotly.express as px, joblib
from pathlib import Path
from src.inference import StudentPerformancePredictor
from src.config import CONFIG

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Predictor")
st.caption("Stacked ensemble (XGBoost + LightGBM + CatBoost + RF) with Optuna tuning & SHAP")

@st.cache_resource
def load():
    return StudentPerformancePredictor("regression"), StudentPerformancePredictor("classification")
REG, CLF = load()

with st.sidebar:
    st.header("Student Profile")
    study = st.slider("Study hours/week", 0, 40, 15)
    attendance = st.slider("Attendance %", 0, 100, 85)
    gpa = st.slider("Prior GPA", 0.0, 4.0, 3.0, 0.05)
    sleep = st.slider("Sleep hours", 3, 11, 7)
    parental = st.selectbox("Parental education (0-4)", [0,1,2,3,4], 2)
    gender = st.selectbox("Gender", ["M", "F", "Other"])
    school = st.selectbox("School type", ["Public", "Private"])
    ses = st.selectbox("Socioeconomic status", ["Low","Mid","High"])
    internet = st.selectbox("Internet access", ["Yes","No"])

rec = dict(study_hours_per_week=study, attendance_rate=attendance, prior_gpa=gpa,
           sleep_hours=sleep, parental_education_level=parental, gender=gender,
           school_type=school, socioeconomic_status=ses, internet_access=internet)

col1, col2 = st.columns(2)
reg = REG.predict([rec])[0]["predicted_grade"]
clf = CLF.predict([rec])[0]
col1.metric("Predicted Final Grade", f"{reg:.1f} / 100")
col2.metric("Pass Probability", f"{clf['pass_probability']:.1%}",
            delta="Pass" if clf["prediction"] else "Fail",
            delta_color="normal" if clf["prediction"] else "inverse")

st.divider()
st.subheader("📊 Metric reports")
reports = Path(CONFIG.paths["reports"])
imgs = list(reports.glob("*.png"))
if imgs:
    cols = st.columns(min(3, len(imgs)))
    for c, p in zip(cols * 5, imgs):
        c.image(str(p), caption=p.stem, use_column_width=True)
else:
    st.info("Run `python -m src.evaluate` to generate report images.")