from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import pandas as pd
import os
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# Try multiple possible paths for the model file
possible_paths = [
    BASE_DIR / "model" / "diabetes_decision_tree.pkl",
    Path("model/diabetes_decision_tree.pkl"),
    Path("Backend/model/diabetes_decision_tree.pkl"),
    Path(os.path.join(os.getcwd(), "model", "diabetes_decision_tree.pkl")),
    Path(os.path.join(os.getcwd(), "Backend", "model", "diabetes_decision_tree.pkl")),
]

model_path = None
for path in possible_paths:
    if path.exists():
        model_path = path
        break

if model_path is None:
    raise FileNotFoundError(
        f"Model file 'diabetes_decision_tree.pkl' not found. "
        f"Tried paths: {[str(p) for p in possible_paths]}. "
        f"Current working directory: {os.getcwd()}. "
        f"BASE_DIR: {BASE_DIR}"
    )

model = joblib.load(model_path)

app = FastAPI(title="Diabetes Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],           
)

class Patient(BaseModel):
    gender: int                     
    age: float
    hypertension: int             
    heart_disease: int             
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    smoking_history_current: int  
    smoking_history_ever: int
    smoking_history_former: int
    smoking_history_never: int
    smoking_history_not_current: int

@app.post("/predict")
def predict_diabetes(patient: Patient):
  
    input_dict = patient.dict()
    input_df = pd.DataFrame([input_dict])
    
  
    input_df["glucose_bmi_ratio"] = input_df["blood_glucose_level"] / (input_df["bmi"] + 1e-6) 
    input_df["age_hba1c"] = input_df["age"] * input_df["HbA1c_level"]
    input_df["bmi_age_interaction"] = input_df["bmi"] * input_df["age"]
    input_df["hba1c_glucose_ratio"] = input_df["HbA1c_level"] / (input_df["blood_glucose_level"] + 1e-6)
    input_df["risk_score"] = input_df["HbA1c_level"] * input_df["blood_glucose_level"] * input_df["age"] / 1000
    
 
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    
    
    input_df = input_df[model.feature_names_in_]
    

    prob = model.predict_proba(input_df)[:,1][0]   
    pred_class = int(model.predict(input_df)[0]) 
    

    if prob < 0.3:
        risk_level = "Low"
    elif prob < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        "prediction": pred_class, 
        "probability": float(prob),
        "risk_level": risk_level,
        "message": f"Diabetes risk is {risk_level} ({prob:.1%} probability)"
    }

@app.get("/features")
def get_expected_features():
    """Endpoint to see what features the model expects"""
    return {
        "expected_features": list(model.feature_names_in_),
        "total_features": len(model.feature_names_in_)
    }