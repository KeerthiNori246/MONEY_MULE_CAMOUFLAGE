"""
create_structure_drift.py
--------------------------
Creates structure-only drift baselines by rewiring transaction-to-merchant
edges in each benign snapshot while keeping ALL node features unchanged.

Three intensity levels:
    low    →  5% of tx-merchant edges rewired per day
    medium → 15% of tx-merchant edges rewired per day
    high   → 30% of tx-merchant edges rewired per day

Reads:
    graphs/baselines/*_benign.gpickle

Outputs:
    graphs/baselines/*_struct_drift_low.gpickle
    graphs/baselines/*_struct_drift_medium.gpickle
    graphs/baselines/*_struct_drift_high.gpickle
    graphs/baselines/struct_drift_summary.csv
"""

import os
import glob
import pickle
import random
import pandas as pd
import networkx as nx
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR = os.path.join(_PROJECT_ROOT, "graphs", "baselines")

# ── intensity levels ─────────────────────────────────────────────────────
INTENSITIES = {
    "low":    0.05,
    "medium": 0.15,
    "high":   0.30,
}


# ── core rewiring function ───────────────────────────────────────────────

def rewire_snapshot(G_original: nx.Graph, rewire_ratio: float, seed: int = 42) -> tuple:
    """
    Rewire a fraction of transaction-to-merchant edges in a snapshot.

    Steps:
      1. Find all tx nodes and all merchant nodes in this snapshot
      2. Collect all tx-merchant edges
      3. Randomly select (rewire_ratio * total) edges as candidates
      4. For each candidate tx node, remove its current merchant edge
         and connect it to a different randomly chosen merchant
      5. All node attributes remain completely unchanged

    Parameters
    ----------
    G_original  : the benign snapshot (not modified in place)
    rewire_ratio: fraction of tx-merchant edges to rewire (0.0 to 1.0)
    seed        : random seed for reproducibility

    Returns
    -------
    G_perturbed : new graph with rewired edges, identical node features
    n_rewired   : number of edges actually rewired
    n_total_tm  : total tx-merchant edges available
    """
    random.seed(seed)

    # deep copy so original is untouched
    G = G_original.copy()

    # collect merchant nodes
    merchant_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "merchant"
    ]

    if len(merchant_nodes) < 2:
        # not enough merchants to rewire — return unchanged
        return G, 0, 0

    # collect all tx-merchant edges
    tx_merchant_edges = []
    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get("node_type", "")
        v_type = G.nodes[v].get("node_type", "")
        if (u_type == "transaction" and v_type == "merchant"):
            tx_merchant_edges.append((u, v, data))
        elif (u_type == "merchant" and v_type == "transaction"):
            tx_merchant_edges.append((v, u, data))   # normalize: (tx, merchant)

    n_total_tm = len(tx_merchant_edges)
    if n_total_tm == 0:
        return G, 0, 0

    # how many to rewire
    n_to_rewire = max(1, int(n_total_tm * rewire_ratio))
    candidates  = random.sample(tx_merchant_edges, n_to_rewire)

    n_rewired = 0
    for tx_node, old_merchant, edge_data in candidates:
        # pick a different merchant
        other_merchants = [m for m in merchant_nodes if m != old_merchant]
        if not other_merchants:
            continue

        new_merchant = random.choice(other_merchants)

        # remove old edge, add new edge (same edge attributes)
        if G.has_edge(tx_node, old_merchant):
            G.remove_edge(tx_node, old_merchant)
        G.add_edge(tx_node, new_merchant, **edge_data)
        n_rewired += 1

    return G, n_rewired, n_total_tm


# ── main pipeline ────────────────────────────────────────────────────────

def create_structure_drift_baselines():
    benign_paths = sorted(glob.glob(os.path.join(BASELINE_DIR, "*_benign.gpickle")))

    if not benign_paths:
        raise FileNotFoundError(
            f"No benign baselines found in {BASELINE_DIR}\n"
            f"Run separate_baselines.py first."
        )

    print("\n" + "=" * 65)
    print("  STRUCTURE-ONLY DRIFT BASELINE CREATION")
    print(f"  Found {len(benign_paths)} benign snapshots")
    print(f"  Intensities: {INTENSITIES}")
    print("=" * 65)

    summary_rows = []

    for intensity_name, ratio in INTENSITIES.items():
        print(f"\n  ── Intensity: {intensity_name.upper()} ({ratio*100:.0f}%) ──")

        for path in tqdm(benign_paths, desc=f"  [{intensity_name}]"):
            date_str = os.path.basename(path).replace("_benign.gpickle", "")

            with open(path, "rb") as f:
                G_benign = pickle.load(f)

            # rewire — seed varies per day so patterns aren't identical across days
            day_seed       = hash(date_str + intensity_name) % (2**32)
            G_perturbed, n_rewired, n_total = rewire_snapshot(
                G_benign, ratio, seed=day_seed
            )

            # verify node features are unchanged
            assert G_perturbed.number_of_nodes() == G_benign.number_of_nodes(), \
                f"Node count changed on {date_str}!"

            # save
            out_name = f"{date_str}_struct_drift_{intensity_name}.gpickle"
            out_path = os.path.join(BASELINE_DIR, out_name)
            with open(out_path, "wb") as f:
                pickle.dump(G_perturbed, f)

            actual_ratio = n_rewired / n_total if n_total > 0 else 0.0
            summary_rows.append({
                "date":           date_str,
                "intensity":      intensity_name,
                "target_ratio":   ratio,
                "n_tx_merch_edges": n_total,
                "n_rewired":      n_rewired,
                "actual_ratio":   round(actual_ratio, 4),
            })

        print(f"  Done — {len(benign_paths)} snapshots saved.")

    # save summary CSV
    df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(BASELINE_DIR, "struct_drift_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n  Summary → {summary_path}")

    # quick stats
    print("\n" + "=" * 65)
    print("  REWIRING STATS")
    print(f"  {'Intensity':<10} {'Avg edges/day':>15} {'Avg rewired':>13} {'Avg ratio':>11}")
    print("  " + "-" * 52)
    for name in INTENSITIES:
        sub = df[df["intensity"] == name]
        print(
            f"  {name:<10} "
            f"{sub['n_tx_merch_edges'].mean():>15.1f} "
            f"{sub['n_rewired'].mean():>13.1f} "
            f"{sub['actual_ratio'].mean():>11.4f}"
        )

    print("\n" + "=" * 65)
    print("  DONE")
    print(f"  Outputs → {BASELINE_DIR}")
    print("  Next: run embed_snapshots_drift.py")
    print("=" * 65)

    return df


if __name__ == "__main__":
    create_structure_drift_baselines()