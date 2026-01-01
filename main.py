# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# 1️⃣ Create synthetic dataset
# -----------------------------
data = pd.DataFrame({
    "hours_studied": [2, 5, 10, 12, 1, 8, 7, 15],
    "attendance": [50, 70, 90, 95, 40, 85, 80, 100],
    "practice_exercises": [1, 3, 5, 6, 0, 4, 3, 7],
    "sleep_hours": [5, 6, 7, 8, 4, 7, 6, 8],
    "target": [0, 0, 1, 1, 0, 1, 1, 1]  # 1=good grade, 0=needs improvement
})

X = data[["hours_studied", "attendance", "practice_exercises", "sleep_hours"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# -----------------------------
# 2️⃣ FastAPI app
# -----------------------------
app = FastAPI(title="Student Grade Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input schema
class StudentInput(BaseModel):
    hours_studied: int
    attendance: int
    practice_exercises: int
    sleep_hours: int

# Human-readable output
def human_readable(pred, prob=None):
    if pred == 1:
        msg = "The student is likely to get a good grade"
    else:
        msg = "The student may need to improve to get a good grade"
    if prob is not None:
        msg += f" ({prob*100:.1f}% confidence)"
    return msg

# Prediction helper
def predict_model(model, features):
    prediction = model.predict(features)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][prediction]
    return human_readable(prediction, prob)

# Logistic Regression endpoint
@app.post("/predict/logistic")
def predict_logistic(data: StudentInput):
    features = [[data.hours_studied, data.attendance, data.practice_exercises, data.sleep_hours]]
    return {"prediction": predict_model(log_model, features)}

# Decision Tree endpoint
@app.post("/predict/tree")
def predict_tree(data: StudentInput):
    features = [[data.hours_studied, data.attendance, data.practice_exercises, data.sleep_hours]]
    return {"prediction": predict_model(tree_model, features)}
