from agents.advanced_demand_agent import predict_demand_lstm
from agents.inventory_agent import inventory_decision
from agents.supplier_agent import select_supplier
from agents.feedback_agent import update_reliability
from llm.llm_helper import ask_llm

print("=== Agentic AI Supply Chain ===")

# STEP 1: Predict demand
demand = predict_demand_lstm()
print("Demand:", demand)

# STEP 2: Inventory decision
decision, reorder = inventory_decision(demand)
print("Inventory:", decision, reorder)

# STEP 3: Supplier selection
supplier, _ = select_supplier()
print("Supplier:", supplier)

# STEP 4: Feedback learning update
update_reliability()

# STEP 5: LLM explanation (ONLY ONCE)
explanation = ask_llm(demand, reorder, supplier)
print("\nLLM Insight:\n", explanation)

