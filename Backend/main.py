from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import pandas as pd
import os


BASE_DIR = Path(__file__).resolve().parent

# Try multiple possible paths for the model file
possible_paths = [
    BASE_DIR / "model" / "diabetes_decision_tree.pkl",  # Relative to main.py
    Path("model/diabetes_decision_tree.pkl"),  # Relative to current directory
    Path("Backend/model/diabetes_decision_tree.pkl"),  # From project root
    Path(os.path.join(os.getcwd(), "model", "diabetes_decision_tree.pkl")),  # From cwd
    Path(os.path.join(os.getcwd(), "Backend", "model", "diabetes_decision_tree.pkl")),  # From cwd/Backend
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


# origins = [
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000", 
#     "https://diabetes-risk-prediction-eight.vercel.app/"
   
# ]

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
   
    input_df = pd.DataFrame([patient.dict()])
    
    
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    
 
    input_df = input_df[model.feature_names_in_]
    
   
    prob = model.predict_proba(input_df)[:,1][0]   
    pred_class = int(model.predict(input_df)[0]) 
    
    return {"prediction": pred_class, "probability": prob}
