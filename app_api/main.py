from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from app_api.schemas import IrisFeatures


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"


app = FastAPI(
    title="Iris Classification API",
    description="A simple API that predicts the Iris flower species.",
    version="1.0.0",
)


try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    target_names = model_data["target_names"]
except FileNotFoundError as error:
    raise RuntimeError(
        f"Model file was not found at: {MODEL_PATH}"
    ) from error


@app.get("/")
def home() -> dict:
    return {
        "message": "Iris Classification API is running"
    }


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(features: IrisFeatures) -> dict:
    try:
        input_data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]]

        prediction_number = int(model.predict(input_data)[0])
        prediction_name = target_names[prediction_number]

        probabilities = model.predict_proba(input_data)[0]

        return {
            "prediction_id": prediction_number,
            "prediction": prediction_name,
            "probabilities": {
                target_names[index]: round(float(probability), 4)
                for index, probability in enumerate(probabilities)
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error