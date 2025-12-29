from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Initialize app
app = FastAPI(title="Iris ML API")

# Enable CORS so frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load trained models
log_model = joblib.load("logistic_model.pkl")
tree_model = joblib.load("decision_tree_model.pkl")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Logistic Regression endpoint
@app.post("/predict/logistic")
def predict_logistic(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = log_model.predict(features)[0]
    return {"prediction": int(prediction)}

# Decision Tree endpoint
@app.post("/predict/tree")
def predict_tree(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = tree_model.predict(features)[0]
    return {"prediction": int(prediction)}
