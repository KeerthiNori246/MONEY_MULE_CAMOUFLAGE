import os
import glob
import pickle
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

SNAPSHOT_DIR = "../../graphs/snapshots"
BASELINE_DIR = "../../graphs/baselines"

os.makedirs(BASELINE_DIR, exist_ok=True)

def separate_baselines():
    """Create benign/adversarial snapshots from existing ones"""
    pattern = f"{SNAPSHOT_DIR}/*.gpickle"
    dates = sorted([os.path.basename(p).replace('.gpickle', '') for p in glob.glob(pattern)])
    
    benign_files = []
    adv_files = []
    
    for date_str in tqdm(dates, desc="Separating baselines"):
        # Load full snapshot
        path = f"{SNAPSHOT_DIR}/{date_str}.gpickle"
        with open(path, 'rb') as f:
            G_full = pickle.load(f)
        
        # Benign: subgraph with ONLY non-fraud transactions
        G_benign = nx.Graph()
        for node, data in G_full.nodes(data=True):
            if data.get("node_type") != "transaction" or data.get("fraud") == 0:
                G_benign.add_node(node, **data)
        
        # Copy benign edges (only connect if both endpoints present)
        for u, v, data in G_full.edges(data=True):
            if u in G_benign and v in G_benign:
                G_benign.add_edge(u, v, **data)
        
        # Adversarial: subgraph with ONLY fraud transactions
        G_adv = nx.Graph()
        for node, data in G_full.nodes(data=True):
            if data.get("node_type") != "transaction" or data.get("fraud") == 1:
                G_adv.add_node(node, **data)
        
        for u, v, data in G_full.edges(data=True):
            if u in G_adv and v in G_adv:
                G_adv.add_edge(u, v, **data)
        
        # Save
        pickle.dump(G_benign, open(f"{BASELINE_DIR}/{date_str}_benign.gpickle", 'wb'))
        pickle.dump(G_adv, open(f"{BASELINE_DIR}/{date_str}_adv.gpickle", 'wb'))
        
        benign_files.append((date_str, G_benign.number_of_nodes(), G_benign.number_of_edges()))
        adv_files.append((date_str, G_adv.number_of_nodes(), G_adv.number_of_edges()))
    
    # Stats summary
    df_benign = pd.DataFrame(benign_files, columns=['date', 'nodes', 'edges'])
    df_adv = pd.DataFrame(adv_files, columns=['date', 'nodes', 'edges'])
    
    print("Benign Stats:\n", df_benign.describe())
    print("Adv Stats:\n", df_adv.describe())
    
    df_benign.to_csv(f"{BASELINE_DIR}/benign_summary.csv")
    df_adv.to_csv(f"{BASELINE_DIR}/adv_summary.csv")
    print(f"✅ Baselines saved to {BASELINE_DIR}/")

if __name__ == "__main__":
    separate_baselines()
