import os
import pandas as pd

print("Feedback Agent Started")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUPPLIERS_PATH = os.path.join(BASE_DIR, "data", "suppliers.csv")
PERFORMANCE_PATH = os.path.join(BASE_DIR, "data", "performance.csv")

print("Reading suppliers from:", SUPPLIERS_PATH)
print("Reading performance from:", PERFORMANCE_PATH)

suppliers = pd.read_csv(SUPPLIERS_PATH)
performance = pd.read_csv(PERFORMANCE_PATH)

# Update supplier reliability based on delay
for i in range(len(performance)):
    supplier_name = performance.loc[i, "supplier"]
    delay = performance.loc[i, "delivery_delay"]

    if delay > 1:
        suppliers.loc[
            suppliers["supplier"] == supplier_name, "reliability"
        ] -= 0.05

# Ensure reliability stays >= 0
suppliers["reliability"] = suppliers["reliability"].clip(lower=0)

print("\nUpdated Supplier Reliability:")
print(suppliers[["supplier", "reliability"]])
