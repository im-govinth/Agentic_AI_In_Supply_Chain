import os
import pandas as pd

print("Inventory Agent Started")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
inventory_path = os.path.join(BASE_DIR, "data", "inventory.csv")

print("Reading inventory from:", inventory_path)

inventory = pd.read_csv(inventory_path)

predicted_demand = 580

current_stock = inventory.loc[0, "current_stock"]
reorder_level = inventory.loc[0, "reorder_level"]

print("Current Stock:", current_stock)
print("Predicted Demand:", predicted_demand)

if current_stock < reorder_level:
    reorder_quantity = predicted_demand - current_stock
    print("Decision: Reorder stock")
    print("Reorder Quantity:", reorder_quantity)
else:
    print("Decision: No reorder needed")
