import ollama

def ask_llm(demand, reorder, supplier):

    prompt = f"""
You are an AI assistant inside a supply chain management system.

The system already made these decisions using AI models.

Actual system output:
- Predicted demand: {demand}
- Inventory reorder quantity: {reorder}
- Selected supplier: {supplier}

Your task:
Explain these results clearly.

Rules:
- Do NOT invent formulas.
- Do NOT mention reorder point.
- Do NOT mention EOQ.
- Do NOT assume lead time.
- ONLY explain what demand agent, inventory agent, and supplier agent did.

Explain in simple professional terms.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
