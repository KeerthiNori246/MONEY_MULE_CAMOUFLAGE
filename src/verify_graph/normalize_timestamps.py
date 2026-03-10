from load_graph import load_graph
from datetime import datetime
import pickle
import networkx as nx

def normalize_timestamps(G):
    """Convert DD-MM-YYYY to YYYY-MM-DD for all transaction nodes"""
    updated = 0
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "transaction":
            date = data.get("date")
            if not date:
                continue
                
            try:
                # Parse DD-MM-YYYY (your format: '23-01-2025')
                old_date = datetime.strptime(date, "%d-%m-%Y")
                # Convert to YYYY-MM-DD standard
                new_date = old_date.strftime("%Y-%m-%d")
                G.nodes[node]["date"] = new_date
                updated += 1
                print(f"Fixed: {date} -> {new_date}")
            except ValueError:
                # Already in YYYY-MM-DD or invalid, skip
                continue
    
    print(f"Updated {updated} timestamps")
    return G
