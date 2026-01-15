import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Student Details (MANDATORY)
# -----------------------------
NAME = "KULVANTH SHOURY"
ROLL_NO = "2022BCS0217"

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("output", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("dataset/winequality.csv", sep=";")

# -----------------------------
# Features & Target
# -----------------------------
X = data.drop("quality", axis=1)
y = data["quality"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Preprocessing (Scaling)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# Print metrics (REQUIRED)
# -----------------------------
print(f"Name: {NAME}")
print(f"Roll No: {ROLL_NO}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "output/model.pkl")

# -----------------------------
# Save results to JSON
# -----------------------------
results = {
    "name": NAME,
    "roll_no": ROLL_NO,
    "mse": mse,
    "r2_score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

