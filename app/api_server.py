# app/api_server.py
import mlflow
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import os 

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Capture INFO level messages and above (WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # The format for each log line
    filename='logs/api.log' # The destination file
)

# Define the input schema using Pydantic for validation
class FlowerFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(title="Iris Model Serving API", version="1.0")
Instrumentator().instrument(app).expose(app)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load the registered model from the MLflow Model Registry
MODEL_NAME = "iris-classifier-unique"
MODEL_STAGE = "Production" # Change to "None" if you haven't transitioned the model

logging.info(f"Using MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
logging.info(f"Loading model '{MODEL_NAME}' version '{MODEL_STAGE}'...")
try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e} {MLFLOW_TRACKING_URI}")
    model = None # Set model to None if loading fails

@app.post("/predict")
def predict(features: FlowerFeatures):
    if model is None:
        return {"error": "Model is not available, please check the logs."}, 503

    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)
    predicted_class_id = int(prediction[0])

    class_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = class_mapping.get(predicted_class_id, "Unknown")

    logging.info(f"Prediction successful. Input: {features.dict()}, Output: {predicted_species}")
    return {"predicted_class_id": predicted_class_id, "predicted_species": predicted_species}

@app.get("/health")
def health_check():
    return {"status": "ok"}