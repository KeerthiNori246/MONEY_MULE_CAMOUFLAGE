from load_graph import load_graph
from datetime import datetime
import pickle

GRAPH_PATH = "../../graphs/transaction_graph.gpickle"

G = load_graph()
fixed = 0

print("Checking current dates...")
for node, data in G.nodes(data=True):
    if data.get("node_type") == "transaction":
        date = data["date"]
        print(f"Sample: '{date}'")  # Shows ACTUAL format
        break

print("Fixing DD-MM-YYYY → YYYY-MM-DD...")
for node, data in G.nodes(data=True):
    if data.get("node_type") == "transaction":
        date = data["date"]
        try:
            # Convert '23-01-2025' → '2025-01-23'
            parsed = datetime.strptime(date, "%d-%m-%Y")
            G.nodes[node]["date"] = parsed.strftime("%Y-%m-%d")
            fixed += 1
        except:
            pass  # Already good or unparseable

print(f"Fixed {fixed} dates")

# OVERWRITE your graph
with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)

print("✅ GRAPH FIXED! Now run: python sci_checks.py")
