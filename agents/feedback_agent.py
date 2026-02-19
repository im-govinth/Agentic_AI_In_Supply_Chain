import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERFORMANCE_PATH = os.path.join(BASE_DIR, "data", "performance.csv")
SUPPLIER_PATH = os.path.join(BASE_DIR, "data", "suppliers.csv")


def update_reliability():

    perf = pd.read_csv(PERFORMANCE_PATH)
    suppliers = pd.read_csv(SUPPLIER_PATH)

    for _, row in perf.iterrows():

        supplier = row["supplier"]
        delay = row["delivery_delay"]
        quality_issue = row["quality_issue"]

        reward = 1

        if delay > 1:
            reward -= 0.5

        if quality_issue == 1:
            reward -= 0.5

        suppliers.loc[
            suppliers["supplier"] == supplier,
            "reliability"
        ] += reward * 0.02

    suppliers["reliability"] = suppliers["reliability"].clip(0,1)

    suppliers.to_csv(SUPPLIER_PATH, index=False)

    return suppliers
