# Script used fastapi to exopose the trained model as web api endpoint

import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = "ML Model Inference API")

# Load the model artfact at the startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return{"status": "Model API is online and healthy"}

@app.post("/predict")
def predict(features: IrisFeatures):
    payload = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = int(model.predict(payload)[0])
    return {"prediction": prediction}
