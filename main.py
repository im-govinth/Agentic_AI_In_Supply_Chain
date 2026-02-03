import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("\n--- AGENTIC AI IN SUPPLY CHAIN MANAGEMENT ---\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# 1. DEMAND FORECASTING AGENT
# ===============================
sales_path = os.path.join(BASE_DIR, "data", "sales.csv")
print("Reading sales data from:", sales_path)

sales = pd.read_csv(sales_path)
sales["month_index"] = range(1, len(sales) + 1)

X = sales[["month_index"]]
y = sales["sales"]

model = LinearRegression()
model.fit(X, y)

next_month = np.array([[len(sales) + 1]])
predicted_demand = int(model.predict(next_month)[0])

print("Predicted Demand:", predicted_demand)

# ===============================
# 2. INVENTORY MANAGEMENT AGENT
# ===============================
inventory_path = os.path.join(BASE_DIR, "data", "inventory.csv")
print("Reading inventory data from:", inventory_path)

inventory = pd.read_csv(inventory_path)

current_stock = inventory.loc[0, "current_stock"]
reorder_level = inventory.loc[0, "reorder_level"]

if current_stock < reorder_level:
    reorder_quantity = predicted_demand - current_stock
    inventory_decision = "Reorder"
else:
    reorder_quantity = 0
    inventory_decision = "No Reorder"

print("Inventory Decision:", inventory_decision)
print("Reorder Quantity:", reorder_quantity)

# ===============================
# 3. SUPPLIER SELECTION AGENT
# ===============================
suppliers_path = os.path.join(BASE_DIR, "data", "suppliers.csv")
print("Reading suppliers data from:", suppliers_path)

suppliers = pd.read_csv(suppliers_path)

suppliers["score"] = (
    0.5 * suppliers["cost"]
    + 0.3 * suppliers["delivery_time"]
    - 0.2 * suppliers["reliability"]
)

best_supplier = suppliers.loc[suppliers["score"].idxmin()]
print("Selected Supplier:", best_supplier["supplier"])

# ===============================
# 4. FEEDBACK & LEARNING AGENT
# ===============================
performance_path = os.path.join(BASE_DIR, "data", "performance.csv")
print("Reading performance data from:", performance_path)

performance = pd.read_csv(performance_path)

for i in range(len(performance)):
    supplier_name = performance.loc[i, "supplier"]
    delay = performance.loc[i, "delivery_delay"]

    if delay > 1:
        suppliers.loc[
            suppliers["supplier"] == supplier_name, "reliability"
        ] -= 0.05

suppliers["reliability"] = suppliers["reliability"].clip(lower=0)

print("\nUpdated Supplier Reliability:")
print(suppliers[["supplier", "reliability"]])

print("\n--- SUPPLY CHAIN DECISION CYCLE COMPLETED ---")
print("MAIN FILE IS RUNNING")
