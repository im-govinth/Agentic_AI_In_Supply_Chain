import sys
import os

# Fix path so agents can be imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from agents.advanced_demand_agent import predict_demand_lstm
from agents.inventory_agent import inventory_decision
from agents.supplier_agent import select_supplier
from llm.llm_helper import ask_llm


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Agentic AI Supply Chain",
    layout="wide"
)

st.title("Agentic AI Supply Chain Dashboard")

st.write("Real-time AI powered supply chain optimization system")

# -----------------------------
# LOAD DATA
# -----------------------------
sales_path = os.path.join(BASE_DIR, "data", "sales.csv")
suppliers_path = os.path.join(BASE_DIR, "data", "suppliers.csv")

sales_df = pd.read_csv(sales_path)
suppliers_df = pd.read_csv(suppliers_path)

# -----------------------------
# RUN AI SYSTEM
# -----------------------------
if st.button("Run Agentic AI System"):

    demand = predict_demand_lstm()
    decision, reorder = inventory_decision(demand)
    supplier, suppliers_df = select_supplier()

    explanation = ask_llm(demand, reorder, supplier)

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Demand", demand)
    col2.metric("Reorder Quantity", reorder)
    col3.metric("Selected Supplier", supplier)

    st.success("AI decision completed")

    # -----------------------------
    # DEMAND GRAPH
    # -----------------------------
    st.subheader("Demand Forecast Visualization")

    fig, ax = plt.subplots()

    ax.plot(sales_df["sales"], marker='o')
    ax.axhline(y=demand, color='r', linestyle='--')

    ax.set_title("Historical Sales and Predicted Demand")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -----------------------------
    # SUPPLIER COMPARISON
    # -----------------------------
    st.subheader("Supplier Comparison")

    fig2, ax2 = plt.subplots()

    ax2.bar(suppliers_df["supplier"], suppliers_df["cost"])

    ax2.set_title("Supplier Cost Comparison")
    ax2.set_xlabel("Supplier")
    ax2.set_ylabel("Cost")

    st.pyplot(fig2)

    # -----------------------------
    # SHOW DATA TABLE
    # -----------------------------
    st.subheader("Supplier Data")

    st.dataframe(suppliers_df)

    # -----------------------------
    # LLM EXPLANATION
    # -----------------------------
    st.subheader("AI Explanation")

    st.info(explanation)


# -----------------------------
# SHOW RAW DATA
# -----------------------------
st.subheader("Historical Sales Data")
st.dataframe(sales_df)

