# Diabetes Risk Prediction — Full Project

A complete diabetes risk prediction app with:

- Backend: FastAPI API serving a trained model (`Backend/main.py` + `Backend/model/diabetes_pipeline.pkl`)
- Frontend: Next.js (TypeScript) UI in `frontend/` for collecting inputs and displaying predictions
- Dataset: `diabetes_prediction_dataset 3.csv` (used during development/training)

Model performance: the exported model reports 88.8% accuracy on the evaluation dataset.

## Project structure

- `Backend/` — FastAPI app, Python dependencies, and the serialized model pipeline
- `frontend/` — Next.js application (app router) for the user interface
- `main.ipynb` — notebook  used for exploratory analysis/training
- `diabetes_prediction_dataset 3.csv` — dataset file

## Prerequisites

- Python 3.8+ (3.12 compatible)
- Node.js 18+ and npm

## Backend setup and run

From the project root:

```zsh
cd Backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API base URL: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- CORS allows `http://localhost:3000` and `http://127.0.0.1:3000`

### API — POST /predict

Predicts diabetes risk and returns:

- `prediction` (int): 0 = low risk, 1 = high risk
- `probability` (float): probability for the positive class

Request body (JSON):

- `gender` (int)
- `age` (float)
- `hypertension` (int) — 0/1
- `heart_disease` (int) — 0/1
- `bmi` (float)
- `HbA1c_level` (float)
- `blood_glucose_level` (float)
- `smoking_history_current` (int) — 0/1
- `smoking_history_ever` (int) — 0/1
- `smoking_history_former` (int) — 0/1
- `smoking_history_never` (int) — 0/1
- `smoking_history_not_current` (int) — 0/1

Note: smoking history is one-hot encoded; typically one flag should be 1.

Example request:

```json
{
  "gender": 1,
  "age": 45,
  "hypertension": 0,
  "heart_disease": 0,
  "bmi": 27.5,
  "HbA1c_level": 5.7,
  "blood_glucose_level": 110,
  "smoking_history_current": 0,
  "smoking_history_ever": 1,
  "smoking_history_former": 0,
  "smoking_history_never": 0,
  "smoking_history_not_current": 0
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.71
}
```

## Frontend setup and run

From the project root:

```zsh
cd frontend
npm install
npm run dev
```

- App runs at `http://localhost:3000`
- The frontend sends requests to `http://localhost:8000/predict` by default; ensure the backend is running.

### Frontend notes

- Entry pages: `frontend/app/page.tsx` and `frontend/app/Home/page.tsx`
- TypeScript and ESLint are configured (`tsconfig.json`, `eslint.config.mjs`)
- Static assets in `frontend/public/`

## Data and model

- Dataset: `diabetes_prediction_dataset 3.csv` used for development/training.
- Model: `Backend/model/diabetes_pipeline.pkl` (loaded via `joblib`). The backend aligns request features to `model.feature_names_in_` and fills any missing features with `0`.
- Reported accuracy: 88.8%.

## Environment/configuration

- Backend ports and hosts can be changed via `uvicorn` flags.
- If you deploy the frontend, update CORS in `Backend/main.py` (`allow_origins`) with the deployed URL.

## Troubleshooting

- Model load errors: verify `Backend/model/diabetes_pipeline.pkl` exists and versions in `requirements.txt` match the pipeline.
- CORS errors: ensure the frontend origin is allowed in `Backend/main.py` or adjust `allow_origins`.
- Incorrect predictions: confirm categorical encodings (e.g., `gender`) and one-hot flags are set correctly.

## Future improvements

- Document exact categorical encodings and valid ranges.
- Add unit/integration tests for both backend and frontend.
- Add CI to run lint, type-check, and minimal tests.

## Files

- `Backend/main.py` — FastAPI app exposing `/predict`
- `Backend/requirements.txt` — backend Python dependencies
- `Backend/model/diabetes_pipeline.pkl` — trained pipeline
- `frontend/` — Next.js client app
- `main.ipynb` — notebook used during development
- `diabetes_prediction_dataset 3.csv` — dataset



