import os
import pandas as pd

print("Supplier Agent Started")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPLIERS_PATH = os.path.join(BASE_DIR, "data", "suppliers.csv")

print("Reading suppliers from:", SUPPLIERS_PATH)

suppliers = pd.read_csv(SUPPLIERS_PATH)

cost_weight = 0.5
delivery_weight = 0.3
reliability_weight = 0.2

suppliers["score"] = (
    cost_weight * suppliers["cost"]
    + delivery_weight * suppliers["delivery_time"]
    - reliability_weight * suppliers["reliability"]
)

best_supplier = suppliers.loc[suppliers["score"].idxmin()]

print("Selected Supplier:")
print(best_supplier)
