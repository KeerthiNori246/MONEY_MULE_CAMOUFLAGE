from load_graph import load_graph
from datetime import datetime
import pickle

GRAPH_PATH = "../../graphs/transaction_graph.gpickle"

def force_normalize(G):
    fixed = 0
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "transaction":
            date = data["date"]
            try:
                # Your format: '23-01-2025' → '2025-01-23'
                parsed = datetime.strptime(date, "%d-%m-%Y")
                G.nodes[node]["date"] = parsed.strftime("%Y-%m-%d")
                fixed += 1
            except:
                print(f"⚠️ Skip: {date}")
    print(f"✅ Fixed {fixed} dates")
    return fixed

G = load_graph()
fixed = force_normalize(G)
with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)
print("SAVED! Run: python sci_checks.py")
