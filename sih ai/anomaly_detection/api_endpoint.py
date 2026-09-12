from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn
from feature_engineering import engineer_features
from explainability_engine import ExplainabilityEngine

app = FastAPI(
    title="SIH Land Acquisition Anomaly Detection API",
    description="Real-time inference API for flagging anomalous land acquisition records.",
    version="1.0"
)

# Pydantic schema for request validation
class LandRecordRequest(BaseModel):
    record_id: str
    state: str
    district: str
    land_type: str
    project_type: str
    cadastral_area_sqm: float
    survey_area_sqm: float
    compensation_amount: float
    displaced_families: int
    notification_date: str
    award_date: str
    rr_disbursement_ratio: float

# Global resources
preprocessor = None
iforest = None
lof = None
explainer = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.on_event("startup")
def load_models():
    global preprocessor, iforest, lof, explainer
    print("Loading models into memory...")
    try:
        preprocessor = joblib.load(os.path.join(BASE_DIR, "models/preprocessor.pkl"))
        iforest = joblib.load(os.path.join(BASE_DIR, "models/iforest_model.pkl"))
        lof = joblib.load(os.path.join(BASE_DIR, "models/lof_model.pkl"))
        explainer = ExplainabilityEngine()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.post("/predict")
def predict_anomaly(record: LandRecordRequest):
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Convert request to DataFrame
    df_raw = pd.DataFrame([record.model_dump()])
    
    # Engineer features
    df_engineered = engineer_features(df_raw)
    
    # Preprocess
    X = preprocessor.transform(df_engineered)
    
    # Predict with iForest and LOF
    # In sklearn iForest/LOF, smaller scores indicate more anomalous. 
    # We invert it so higher = more anomalous
    score_iforest = -iforest.score_samples(X)[0]
    score_lof = -lof.score_samples(X)[0]
    
    # Crude normalization for this specific dataset based on typical ranges
    # In production, use standard scalers fit on the scores during training
    norm_iforest = min(max((score_iforest - (-0.6)) / 0.3, 0.0), 1.0) 
    
    anomaly_score = float(norm_iforest) # Primary score from iForest
    
    # Determine Risk Level
    is_anomaly = anomaly_score > 0.65
    
    if anomaly_score < 0.4:
        risk_level = "Low"
    elif anomaly_score < 0.65:
        risk_level = "Medium"
    elif anomaly_score < 0.85:
        risk_level = "High"
    else:
        risk_level = "Critical"
        
    # Get Explanations if anomalous
    top_risk_factors = []
    if risk_level in ["High", "Critical"]:
        top_risk_factors = explainer.explain_instance(df_engineered)
        
    return {
        "record_id": record.record_id,
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 4),
        "risk_level": risk_level,
        "top_risk_factors": top_risk_factors
    }

if __name__ == "__main__":
    # Run server locally
    uvicorn.run("api_endpoint:app", host="0.0.0.0", port=8000, reload=True)
