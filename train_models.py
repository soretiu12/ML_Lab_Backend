import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("iris_binary.csv")

# Features & target
X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_model.predict(X_test)))

# Train Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_model.predict(X_test)))

# Save models
joblib.dump(log_model, "logistic_model.pkl")
joblib.dump(tree_model, "decision_tree_model.pkl")
print("Models trained and saved.")
