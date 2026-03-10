import os
import glob
import pickle
from datetime import datetime
from collections import defaultdict
import networkx as nx
from tqdm import tqdm
from load_graph import load_graph

SNAPSHOT_DIR = "../../graphs/snapshots"

def create_daily_snapshots(G):
    print("🗑️ Deleting broken snapshots...")
    # FIX: Use Python, not bash
    for f in glob.glob(f"{SNAPSHOT_DIR}/*.gpickle"):
        os.remove(f)
        print(f"Deleted: {f}")
    
    print("Creating CORRECT daily snapshots...")
    
    # Extract transactions with timestamps
    transactions = []
    for node, data in tqdm(G.nodes(data=True), desc="Extracting tx"):
        if data.get("node_type") == "transaction":
            ts_str = f"{data['date']} {data['time']}"
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            transactions.append((node, ts, data))
    
    transactions.sort(key=lambda x: x[1])
    dates = [t[1].date() for t in transactions]
    min_date, max_date = min(dates), max(dates)
    
    print(f"📅 Range: {min_date} → {max_date} ({len(transactions)} total tx)")
    
    # Group transactions by day
    daily_tx = defaultdict(list)
    for node, ts, data in transactions:
        date_str = ts.strftime("%Y-%m-%d")
        daily_tx[date_str].append((node, data))
    
    # Create TRUE daily snapshots
    for date_str, day_tx in tqdm(daily_tx.items(), desc="Snapshots"):
        snapshot = nx.Graph()
        
        # Add ALL persistent nodes (customers/merchants/devices/locations)
        for node, data in G.nodes(data=True):
            if data.get("node_type") != "transaction":
                snapshot.add_node(node, **data)
        
        # Add ONLY today's transactions + their edges
        day_edges = 0
        for tx_node, tx_data in day_tx:
            snapshot.add_node(tx_node, **tx_data)
            
            # Connect ONLY transactions that happened TODAY
            for neighbor in G.neighbors(tx_node):
                snapshot.add_edge(neighbor, tx_node, **G[neighbor][tx_node])
                day_edges += 1
        
        save_snapshot(snapshot, date_str, len(day_tx), day_edges)
    
    print(f"✅ {len(daily_tx)} VALID snapshots created!")
    return daily_tx

def save_snapshot(G, date_str, tx_count, edge_count):
    path = f"{SNAPSHOT_DIR}/{date_str}.gpickle"
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"💾 {date_str}: {tx_count} tx, {edge_count} edges, {G.number_of_nodes()} nodes")

if __name__ == "__main__":
    G = load_graph()
    create_daily_snapshots(G)
