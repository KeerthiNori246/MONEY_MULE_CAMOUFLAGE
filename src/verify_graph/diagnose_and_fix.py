from load_graph import load_graph
import pickle
from datetime import datetime

G = load_graph()
GRAPH_PATH = "../../graphs/transaction_graph.gpickle"

# DIAGNOSE - print EXACT format
print("CURRENT DATE FORMAT:")
for node, data in G.nodes(data=True):
    if data.get("node_type") == "transaction":
        print(f"Date: '{data['date']}' | Time: '{data['time']}' | Type: {type(data['date'])}")
        break

# FORCE YYYY-MM-DD regardless of input
fixed = 0
for node, data in G.nodes(data=True):
    if data.get("node_type") == "transaction":
        date_str = str(data["date"]).strip()
        time_str = str(data["time"]).strip()
        
        # Nuclear option: ANY string → YYYY-MM-DD (pick first valid)
        for fmt in ["%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                parsed = datetime.strptime(date_str, fmt)
                G.nodes[node]["date"] = parsed.strftime("%Y-%m-%d")
                G.nodes[node]["time"] = time_str[:8]  # Ensure HH:MM:SS
                fixed += 1
                break
            except:
                continue

print(f"Fixed {fixed} nodes")
with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)
print("✅ FIXED. Run sci_checks.py")
