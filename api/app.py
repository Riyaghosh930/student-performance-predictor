"""REST inference API with pydantic validation."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference import StudentPerformancePredictor

app = FastAPI(title="Student Performance Predictor API", version="1.0")
REG = StudentPerformancePredictor("regression")
CLF = StudentPerformancePredictor("classification")


class StudentFeatures(BaseModel):
    study_hours_per_week: float = Field(..., ge=0, le=80)
    attendance_rate: float = Field(..., ge=0, le=100)
    prior_gpa: float = Field(..., ge=0, le=4)
    sleep_hours: float = Field(..., ge=0, le=16)
    parental_education_level: int = Field(..., ge=0, le=4)
    gender: str
    school_type: str
    socioeconomic_status: str
    internet_access: str


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/predict/grade")
def predict_grade(payload: StudentFeatures):
    try:
        return REG.predict([payload.model_dump()])[0]
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/predict/pass")
def predict_pass(payload: StudentFeatures):
    try:
        return CLF.predict([payload.model_dump()])[0]
    except Exception as e:
        raise HTTPException(400, str(e))