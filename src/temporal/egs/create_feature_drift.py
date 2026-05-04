"""
create_feature_drift.py
------------------------
Creates feature-only drift baselines by perturbing transaction amount
values in each benign snapshot while keeping ALL edges identical.

Three intensity levels:
    low    →  5% Gaussian noise on amount
    medium → 15% Gaussian noise on amount
    high   → 30% Gaussian noise on amount

Graph topology (every edge, every connection) is byte-for-byte identical.
Only transaction node 'amount' changes.

Reads:
    graphs/baselines/*_benign.gpickle

Outputs:
    graphs/baselines/*_feature_drift_low.gpickle
    graphs/baselines/*_feature_drift_medium.gpickle
    graphs/baselines/*_feature_drift_high.gpickle
    graphs/baselines/feature_drift_summary.csv
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR = os.path.join(_PROJECT_ROOT, "graphs", "baselines")

INTENSITIES  = {"low": 0.05, "medium": 0.15, "high": 0.30}
AMOUNT_FLOOR = 1.0   # clip floor — amounts must stay positive


# ── perturbation ─────────────────────────────────────────────────────────

def perturb_features(G_original: nx.Graph, noise_std: float, seed: int) -> tuple:
    """
    Perturb transaction node 'amount' with multiplicative Gaussian noise.
    new_amount = original_amount * (1 + N(0, noise_std))
    Clipped to AMOUNT_FLOOR. All edges stay identical.
    """
    rng = np.random.default_rng(seed)
    G   = G_original.copy()

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

    # sanity check — topology must be unchanged
    assert G.number_of_edges() == G_original.number_of_edges()
    assert G.number_of_nodes() == G_original.number_of_nodes()

    stats = {
        "orig_mean":  float(orig_amounts.mean()),
        "new_mean":   float(new_amounts.mean()),
        "pct_change": float(abs(new_amounts.mean() - orig_amounts.mean())
                            / (orig_amounts.mean() + 1e-8) * 100),
    }
    return G, len(tx_nodes), stats


# ── main ─────────────────────────────────────────────────────────────────

def create_feature_drift_baselines():
    benign_paths = sorted(glob.glob(os.path.join(BASELINE_DIR, "*_benign.gpickle")))

    if not benign_paths:
        raise FileNotFoundError(
            f"No benign baselines found in {BASELINE_DIR}\n"
            "Run separate_baselines.py first."
        )

    print("\n" + "=" * 65)
    print("  FEATURE-ONLY DRIFT BASELINE CREATION")
    print(f"  Found {len(benign_paths)} benign snapshots")
    print(f"  Perturbation: amount x (1 + N(0,sigma)) — topology unchanged")
    print("=" * 65)

    summary_rows = []

    for intensity_name, noise_std in INTENSITIES.items():
        print(f"\n  -- Intensity: {intensity_name.upper()} (sigma={noise_std}) --")

        for path in tqdm(benign_paths, desc=f"  [{intensity_name}]"):
            date_str = os.path.basename(path).replace("_benign.gpickle", "")
            day_seed = hash(date_str + intensity_name) % (2**32)

            with open(path, "rb") as f:
                G_benign = pickle.load(f)

            G_perturbed, n_tx, stats = perturb_features(G_benign, noise_std, day_seed)

            out_path = os.path.join(BASELINE_DIR,
                                    f"{date_str}_feature_drift_{intensity_name}.gpickle")
            with open(out_path, "wb") as f:
                pickle.dump(G_perturbed, f)

            summary_rows.append({
                "date":          date_str,
                "intensity":     intensity_name,
                "noise_std":     noise_std,
                "n_tx":          n_tx,
                "orig_mean_amt": round(stats.get("orig_mean", 0), 2),
                "new_mean_amt":  round(stats.get("new_mean",  0), 2),
                "pct_change":    round(stats.get("pct_change", 0), 4),
            })

        print(f"  Done -- {len(benign_paths)} snapshots saved.")

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(BASELINE_DIR, "feature_drift_summary.csv"), index=False)

    print("\n" + "=" * 65)
    print("  PERTURBATION STATS")
    print(f"  {'Intensity':<10} {'Avg tx/day':>12} {'Avg orig amt':>14} "
          f"{'Avg new amt':>13} {'Avg % chg':>11}")
    print("  " + "-" * 55)
    for name in INTENSITIES:
        sub = df[df["intensity"] == name]
        print(f"  {name:<10} {sub['n_tx'].mean():>12.1f} "
              f"{sub['orig_mean_amt'].mean():>14.2f} "
              f"{sub['new_mean_amt'].mean():>13.2f} "
              f"{sub['pct_change'].mean():>10.2f}%")

    print("\n" + "=" * 65)
    print("  DONE")
    print(f"  Outputs -> {BASELINE_DIR}")
    print("  Next: run embed_snapshots_feature_drift.py")
    print("=" * 65)
    return df


if __name__ == "__main__":
    create_feature_drift_baselines()