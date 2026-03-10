from load_graph import load_graph
from collections import Counter
from datetime import datetime
import numpy as np

def check_timestamps(G):
    print("\n--- TIMESTAMP CHECK ---")
    timestamps = []
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "transaction":
            date = data["date"]
            time = data["time"]
            try:
                # NOW CORRECT for your normalized data
                ts = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
                timestamps.append(ts)
            except Exception as e:
                print(f"Parse error: {date} {time} -> {e}")
                continue
    
    print("Valid timestamps:", len(timestamps))
    if timestamps:
        print("✅ SUCCESS - Earliest:", min(timestamps))
        print("✅ SUCCESS - Latest:", max(timestamps))
    else:
        print("No valid timestamps found")



def check_hubs(G):

    print("\n--- HUB DISTRIBUTION CHECK ---")

    degrees = [d for n,d in G.degree()]

    print("Max degree:", max(degrees))
    print("Mean degree:", np.mean(degrees))
    print("Median degree:", np.median(degrees))

    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]

    print("\nTop 10 hubs:")

    for node,deg in top_nodes:

        print(node,deg)


def check_fraud_mixing(G):

    print("\n--- HETEROPHILY CHECK ---")

    fraud_edges = 0
    total_edges = 0

    for u,v in G.edges():

        total_edges += 1

        if G.nodes[u].get("fraud")==1 or G.nodes[v].get("fraud")==1:
            fraud_edges += 1

    ratio = fraud_edges / total_edges

    print("Fraud edges:", fraud_edges)
    print("Total edges:", total_edges)
    print("Fraud edge ratio:", ratio)


def check_degree_distribution(G):

    print("\n--- DEGREE DISTRIBUTION ---")

    degrees = [d for n,d in G.degree()]

    hist = Counter(degrees)

    top = sorted(hist.items())[:10]

    print("Sample distribution (degree -> count):")

    for deg,count in top:
        print(deg,count)


def main():
    print("Loading graph...")
    G = load_graph()
    print("Graph loaded")
    
    check_timestamps(G)  # Now safe
    check_hubs(G)
    check_degree_distribution(G)
    check_fraud_mixing(G)

if __name__ == "__main__":
    main()