"""
create_combined_drift.py
-------------------------
Creates combined drift baselines by applying BOTH edge rewiring AND
feature perturbation to each benign snapshot simultaneously.

This represents real-world money mule behavior:
    - Structure change: rewire tx-merchant edges (same as structure drift)
    - Feature change: perturb transaction amounts (same as feature drift)
    - Both applied together on the same snapshot

Three intensity levels (matched exactly to previous experiments):
    low    → 5%  edge rewire + sigma=0.05 amount noise
    medium → 15% edge rewire + sigma=0.15 amount noise
    high   → 30% edge rewire + sigma=0.30 amount noise

Reads:
    graphs/baselines/*_benign.gpickle

Outputs:
    graphs/baselines/*_combined_drift_low.gpickle
    graphs/baselines/*_combined_drift_medium.gpickle
    graphs/baselines/*_combined_drift_high.gpickle
    graphs/baselines/combined_drift_summary.csv
"""

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR = os.path.join(_PROJECT_ROOT, "graphs", "baselines")

# ── intensity levels (matched to structure + feature experiments) ─────────
INTENSITIES = {
    "low":    {"rewire_ratio": 0.05, "noise_std": 0.05},
    "medium": {"rewire_ratio": 0.15, "noise_std": 0.15},
    "high":   {"rewire_ratio": 0.30, "noise_std": 0.30},
}

AMOUNT_FLOOR = 1.0


# ── step 1: rewire edges (from create_structure_drift.py) ─────────────────

def rewire_edges(G: nx.Graph, rewire_ratio: float, rng_seed: int) -> tuple:
    """
    Rewire a fraction of tx-merchant edges.
    All node attributes unchanged.
    Returns (G_rewired, n_rewired, n_total_tm)
    """
    random.seed(rng_seed)

    merchant_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "merchant"
    ]

    if len(merchant_nodes) < 2:
        return G, 0, 0

    tx_merchant_edges = []
    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get("node_type", "")
        v_type = G.nodes[v].get("node_type", "")
        if u_type == "transaction" and v_type == "merchant":
            tx_merchant_edges.append((u, v, data))
        elif u_type == "merchant" and v_type == "transaction":
            tx_merchant_edges.append((v, u, data))

    n_total    = len(tx_merchant_edges)
    if n_total == 0:
        return G, 0, 0

    n_to_rewire = max(1, int(n_total * rewire_ratio))
    candidates  = random.sample(tx_merchant_edges, n_to_rewire)

    n_rewired = 0
    for tx_node, old_merchant, edge_data in candidates:
        other_merchants = [m for m in merchant_nodes if m != old_merchant]
        if not other_merchants:
            continue
        new_merchant = random.choice(other_merchants)
        if G.has_edge(tx_node, old_merchant):
            G.remove_edge(tx_node, old_merchant)
        G.add_edge(tx_node, new_merchant, **edge_data)
        n_rewired += 1

    return G, n_rewired, n_total


# ── step 2: perturb features (from create_feature_drift.py) ──────────────

def perturb_amounts(G: nx.Graph, noise_std: float, rng_seed: int) -> tuple:
    """
    Perturb transaction amount with multiplicative Gaussian noise.
    new_amount = original_amount * (1 + N(0, noise_std))
    Clipped to AMOUNT_FLOOR.
    Returns (G_perturbed, n_tx_perturbed, stats)
    """
    rng = np.random.default_rng(rng_seed)

    tx_nodes     = []
    orig_amounts = []

    for node, data in G.nodes(data=True):
        if data.get("node_type") == "transaction":
            tx_nodes.append(node)
            orig_amounts.append(float(data.get("amount", 0.0)))

    if not tx_nodes:
        return G, 0, {}

    orig_amounts = np.array(orig_amounts)
    noise        = rng.normal(loc=0.0, scale=noise_std, size=len(tx_nodes))
    new_amounts  = np.clip(orig_amounts * (1.0 + noise), AMOUNT_FLOOR, None)

    for node, new_amt in zip(tx_nodes, new_amounts):
        G.nodes[node]["amount"] = float(new_amt)

    stats = {
        "orig_mean":  float(orig_amounts.mean()),
        "new_mean":   float(new_amounts.mean()),
        "pct_change": float(abs(new_amounts.mean() - orig_amounts.mean())
                            / (orig_amounts.mean() + 1e-8) * 100),
    }
    return G, len(tx_nodes), stats


# ── combined perturbation ─────────────────────────────────────────────────

def apply_combined_drift(G_original: nx.Graph,
                         rewire_ratio: float,
                         noise_std: float,
                         seed: int) -> tuple:
    """
    Apply both rewiring and feature perturbation to a single snapshot.

    Order:
        1. Deep copy original
        2. Rewire edges (structure change)
        3. Perturb amounts (feature change)
        4. Verify node/edge counts consistent

    Returns
    -------
    G_combined  : perturbed graph
    stats       : dict summarising both perturbations
    """
    G = G_original.copy()

    # step 1 — structure
    G, n_rewired, n_total_tm = rewire_edges(G, rewire_ratio, rng_seed=seed)

    # step 2 — features (use seed+1 so noise pattern differs from edge seed)
    G, n_tx, amount_stats = perturb_amounts(G, noise_std, rng_seed=seed + 1)

    # sanity checks
    assert G.number_of_nodes() == G_original.number_of_nodes(), \
        "Node count changed!"
    # edge count may differ slightly due to rewiring — that's expected

    stats = {
        "n_tx_merchant_edges": n_total_tm,
        "n_rewired":           n_rewired,
        "actual_rewire_ratio": n_rewired / n_total_tm if n_total_tm > 0 else 0.0,
        "n_tx_perturbed":      n_tx,
        "orig_mean_amt":       round(amount_stats.get("orig_mean", 0), 2),
        "new_mean_amt":        round(amount_stats.get("new_mean",  0), 2),
        "pct_amt_change":      round(amount_stats.get("pct_change", 0), 4),
    }
    return G, stats


# ── main pipeline ────────────────────────────────────────────────────────

def create_combined_drift_baselines():
    benign_paths = sorted(glob.glob(os.path.join(BASELINE_DIR, "*_benign.gpickle")))

    if not benign_paths:
        raise FileNotFoundError(
            f"No benign baselines found in {BASELINE_DIR}\n"
            "Run separate_baselines.py first."
        )

    print("\n" + "=" * 65)
    print("  COMBINED DRIFT BASELINE CREATION")
    print(f"  Found {len(benign_paths)} benign snapshots")
    print("  Applying: edge rewiring + amount perturbation simultaneously")
    print(f"  Intensities: {INTENSITIES}")
    print("=" * 65)

    summary_rows = []

    for intensity_name, params in INTENSITIES.items():
        rewire_ratio = params["rewire_ratio"]
        noise_std    = params["noise_std"]

        print(f"\n  -- Intensity: {intensity_name.upper()} "
              f"(rewire={rewire_ratio*100:.0f}%, sigma={noise_std}) --")

        for path in tqdm(benign_paths, desc=f"  [{intensity_name}]"):
            date_str = os.path.basename(path).replace("_benign.gpickle", "")
            day_seed = hash(date_str + intensity_name) % (2**32)

            with open(path, "rb") as f:
                G_benign = pickle.load(f)

            G_combined, stats = apply_combined_drift(
                G_benign, rewire_ratio, noise_std, seed=day_seed
            )

            out_path = os.path.join(
                BASELINE_DIR,
                f"{date_str}_combined_drift_{intensity_name}.gpickle"
            )
            with open(out_path, "wb") as f:
                pickle.dump(G_combined, f)

            summary_rows.append({
                "date":              date_str,
                "intensity":         intensity_name,
                "rewire_ratio":      rewire_ratio,
                "noise_std":         noise_std,
                **stats,
            })

        print(f"  Done -- {len(benign_paths)} snapshots saved.")

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(BASELINE_DIR, "combined_drift_summary.csv"), index=False)

    print("\n" + "=" * 65)
    print("  COMBINED DRIFT STATS")
    print(f"  {'Intensity':<10} {'Rewire%':>9} {'Avg rewired':>13} "
          f"{'Sigma':>7} {'Avg amt chg':>13}")
    print("  " + "-" * 58)
    for name, params in INTENSITIES.items():
        sub = df[df["intensity"] == name]
        print(f"  {name:<10} "
              f"{params['rewire_ratio']*100:>8.0f}% "
              f"{sub['n_rewired'].mean():>13.1f} "
              f"{params['noise_std']:>7.2f} "
              f"{sub['pct_amt_change'].mean():>12.2f}%")

    print("\n" + "=" * 65)
    print("  DONE")
    print(f"  Outputs -> {BASELINE_DIR}")
    print("  Next: run embed_snapshots_combined_drift.py")
    print("=" * 65)
    return df


if __name__ == "__main__":
    create_combined_drift_baselines()