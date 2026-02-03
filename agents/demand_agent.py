import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build absolute path to sales.csv
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

print("Looking for data at:", DATA_PATH)

# Load sales data
data = pd.read_csv(DATA_PATH)

# Convert months to numbers
data["month_index"] = range(1, len(data) + 1)

X = data[["month_index"]]
y = data["sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next demand
next_month = np.array([[len(data) + 1]])
prediction = model.predict(next_month)

print("Predicted demand for next month:", int(prediction[0]))

