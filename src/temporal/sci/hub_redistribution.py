"""
hub_redistribution.py
----------------------
Iterates over daily baseline snapshots (benign OR adversarial),
computes ΔD_t and ΔC_t per consecutive snapshot pair, then
combines them into a unified HR_t signal via z-score standardization.

Output CSVs:
    graphs/sci_results/hub_timeseries_benign.csv
    graphs/sci_results/hub_timeseries_adv.csv

Columns:
    date | delta_entropy | delta_centrality | HR_t

Design mirrors temporal_motif_evolution.py exactly.
Key difference: loop is STATEFUL — carries G_prev forward at each step.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from compute_hub_signals import compute_hub_signals

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR  = os.path.join(_PROJECT_ROOT, "graphs", "baselines")
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "graphs", "sci_results")

print(f"[paths] BASELINE_DIR = {BASELINE_DIR}")
print(f"[paths] OUTPUT_DIR   = {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_baseline_snapshots(mode="benign"):
    """
    Load all daily baseline snapshots for a given mode.
    Returns list of (date_str, G) sorted by date ascending.
    Mirrors load_baseline_snapshots() in temporal_motif_evolution.py.
    """
    pattern = f"{BASELINE_DIR}/*_{mode}.gpickle"
    paths   = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No {mode} baselines found in {BASELINE_DIR}. "
            "Run separate_baselines.py first."
        )

    result = []
    for path in paths:
        date_str = os.path.basename(path).replace(f"_{mode}.gpickle", "")
        with open(path, "rb") as f:
            G = pickle.load(f)
        result.append((date_str, G))

    return result   # already sorted by date string (YYYY-MM-DD filenames)


def _zscore(arr):
    """
    Z-score standardize a numpy array.
    Returns zeros if std == 0 (flat signal — no variation to standardize).
    """
    arr = np.array(arr, dtype=float)
    mu  = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def compute_hub_timeseries(mode="benign", alpha=0.5, beta=0.5):
    """
    Compute the hub redistribution time-series for one baseline.

    Parameters
    ----------
    mode  : str    "benign" or "adv"
    alpha : float  weight for Z(ΔD_t)   — default 0.5
    beta  : float  weight for Z(ΔC_t)   — default 0.5

    Returns
    -------
    df : pd.DataFrame
        Columns: date | delta_entropy | delta_centrality | HR_t
    """
    print(f"\n{'='*50}")
    print(f"Computing Hub Redistribution Time-Series — {mode.upper()}")
    print(f"  alpha={alpha}  beta={beta}")
    print(f"{'='*50}")

    snapshots = _load_baseline_snapshots(mode)
    print(f"Loaded {len(snapshots)} snapshots")

    records  = []
    G_prev   = None   # stateful: carry previous snapshot forward

    for date_str, G_curr in tqdm(snapshots, desc=f"Hub signals [{mode}]"):

        signals = compute_hub_signals(G_curr, G_prev)

        records.append({
            "date":             date_str,
            "delta_entropy":    signals["delta_entropy"],
            "delta_centrality": signals["delta_centrality"],
        })

        G_prev = G_curr   # advance the window

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # --- Z-score standardization (full time-series, post-loop) ---
    # Done here, not inside the loop, so mean/std are computed over
    # the complete distribution — same reasoning as motif pipeline's diff().
    z_entropy    = _zscore(df["delta_entropy"].values)
    z_centrality = _zscore(df["delta_centrality"].values)

    df["HR_t"] = alpha * z_entropy + beta * z_centrality

    # Diagnostics
    print(f"\nHub time-series shape: {df.shape}")
    print(df[["date", "delta_entropy", "delta_centrality", "HR_t"]].head(10))
    print(f"\nHR_t stats:")
    print(f"  mean : {df['HR_t'].mean():.4f}")
    print(f"  std  : {df['HR_t'].std():.4f}")
    print(f"  max  : {df['HR_t'].max():.4f}")
    print(f"  min  : {df['HR_t'].min():.4f}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"hub_timeseries_{mode}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    return df


def compute_both_hub(alpha=0.5, beta=0.5):
    """
    Run hub redistribution pipeline for both baselines.
    Returns (df_benign, df_adv).
    """
    df_benign = compute_hub_timeseries("benign", alpha=alpha, beta=beta)
    df_adv    = compute_hub_timeseries("adv",    alpha=alpha, beta=beta)
    return df_benign, df_adv


if __name__ == "__main__":
    compute_both_hub()