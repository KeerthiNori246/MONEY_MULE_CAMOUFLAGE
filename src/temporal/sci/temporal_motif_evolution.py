"""
temporal_motif_evolution.py
----------------------------
Iterates over daily baseline snapshots (benign OR adversarial),
counts triangles and 2-hop wedges per day, then computes delta_M_t.

Output: two DataFrames (benign, adv) with columns:
    date | triangles | wedges | delta_triangles | delta_wedges
"""

import os
import glob
import pickle
import pandas as pd
from tqdm import tqdm

from count_motifs import count_all_motifs

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR  = os.path.join(_PROJECT_ROOT, "graphs", "baselines")
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "graphs", "sci_results")

print(f"[paths] BASELINE_DIR = {BASELINE_DIR}")
print(f"[paths] OUTPUT_DIR   = {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_baseline_snapshots(mode="benign"):
    pattern = f"{BASELINE_DIR}/*_{mode}.gpickle"
    paths = sorted(glob.glob(pattern))

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
    return result


def compute_motif_timeseries(mode="benign"):
    print(f"\n{'='*50}")
    print(f"Computing Temporal Motif Evolution — {mode.upper()}")
    print(f"{'='*50}")

    snapshots = load_baseline_snapshots(mode)
    print(f"Loaded {len(snapshots)} snapshots")

    records = []
    for date_str, G_snapshot in tqdm(snapshots, desc=f"Processing {mode}"):
        counts = count_all_motifs(G_snapshot)
        records.append({
            "date":      date_str,
            "triangles": counts["triangles"],
            "wedges":    counts["wedges"],
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    for col in ["triangles", "wedges"]:
        df[f"delta_{col}"] = df[col].diff().abs().fillna(0)

    print(f"\nMotif time-series shape: {df.shape}")
    print(df[["date", "triangles", "wedges",
              "delta_triangles", "delta_wedges"]].head(10))

    out_path = os.path.join(OUTPUT_DIR, f"motif_timeseries_{mode}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    return df


def compute_both():
    df_benign = compute_motif_timeseries("benign")
    df_adv    = compute_motif_timeseries("adv")
    return df_benign, df_adv


if __name__ == "__main__":
    compute_both()