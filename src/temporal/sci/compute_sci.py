"""
compute_sci.py
--------------
Combines Temporal Motif Evolution and Hub Redistribution signals
into the unified Structural Coordination Intensity (SCI_t) signal.

Formula:
    Step 1 — Collapse motif deltas into ΔM_t:
        ΔM_t = 0.5 · Z(delta_triangles) + 0.5 · Z(delta_wedges)

    Step 2 — HR_t already exists from hub_redistribution.py

    Step 3 — Combine into SCI_t:
        SCI_t = 0.5 · Z(ΔM_t) + 0.5 · Z(HR_t)

Z-scoring is computed independently per baseline (benign stats stay
benign, adversarial stats stay adversarial). This keeps each baseline
self-contained and ensures spike thresholds are meaningful relative
to that baseline's own distribution.

Output CSVs:
    graphs/sci_results/sci_timeseries_benign.csv
    graphs/sci_results/sci_timeseries_adv.csv

Columns:
    date | delta_triangles | delta_wedges | delta_entropy | delta_centrality
         | HR_t | delta_M_t | SCI_t
"""

import os
import numpy as np
import pandas as pd

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "graphs", "sci_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _zscore(arr):
    """
    Z-score standardize a numpy array.
    Returns zeros if std == 0 (flat signal — no variation).
    Identical to _zscore() in hub_redistribution.py.
    """
    arr   = np.array(arr, dtype=float)
    mu    = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_sci_timeseries(mode="benign",
                            motif_w_triangles=0.5,
                            motif_w_wedges=0.5,
                            sci_w_motif=0.5,
                            sci_w_hub=0.5):
    """
    Load motif and hub CSVs for a given baseline, compute SCI_t.

    Parameters
    ----------
    mode              : str    "benign" or "adv"
    motif_w_triangles : float  weight for Z(delta_triangles) in ΔM_t
    motif_w_wedges    : float  weight for Z(delta_wedges) in ΔM_t
    sci_w_motif       : float  weight for Z(ΔM_t) in SCI_t
    sci_w_hub         : float  weight for Z(HR_t) in SCI_t

    Returns
    -------
    df : pd.DataFrame
        Full merged time-series with SCI_t column.
    """
    print(f"\n{'='*50}")
    print(f"Computing SCI_t — {mode.upper()}")
    print(f"  ΔM_t  weights: triangles={motif_w_triangles}  wedges={motif_w_wedges}")
    print(f"  SCI_t weights: motif={sci_w_motif}  hub={sci_w_hub}")
    print(f"{'='*50}")

    # --- Load CSVs ---
    motif_path = os.path.join(OUTPUT_DIR, f"motif_timeseries_{mode}.csv")
    hub_path   = os.path.join(OUTPUT_DIR, f"hub_timeseries_{mode}.csv")

    if not os.path.exists(motif_path):
        raise FileNotFoundError(
            f"Motif CSV not found: {motif_path}\n"
            "Run temporal_motif_evolution.py first (Phase 1)."
        )
    if not os.path.exists(hub_path):
        raise FileNotFoundError(
            f"Hub CSV not found: {hub_path}\n"
            "Run hub_redistribution.py first (Phase 3)."
        )

    df_motif = pd.read_csv(motif_path)
    df_hub   = pd.read_csv(hub_path)

    # --- Merge on date ---
    df = pd.merge(df_motif, df_hub, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"Merge produced empty DataFrame for mode={mode}. "
            "Check that motif and hub CSVs cover the same date range."
        )

    print(f"Merged shape: {df.shape}  ({len(df)} days)")

    # --- Step 1: Collapse motif deltas → ΔM_t ---
    # Z-score each motif delta independently (within this baseline only)
    z_triangles = _zscore(df["delta_triangles"].values)
    z_wedges    = _zscore(df["delta_wedges"].values)

    df["delta_M_t"] = motif_w_triangles * z_triangles + motif_w_wedges * z_wedges

    # --- Step 2: HR_t already exists in hub CSV — already z-scored internally ---
    # We re-Z-score it here at the SCI level for consistent combination.
    # This is correct: HR_t's internal z-score normalized its own components,
    # but SCI needs ΔM_t and HR_t on the same scale relative to each other.
    z_delta_M = _zscore(df["delta_M_t"].values)
    z_HR      = _zscore(df["HR_t"].values)

    # --- Step 3: SCI_t ---
    df["SCI_t"] = sci_w_motif * z_delta_M + sci_w_hub * z_HR

    # Diagnostics
    print(f"\nSCI_t stats:")
    print(f"  mean : {df['SCI_t'].mean():.4f}")
    print(f"  std  : {df['SCI_t'].std():.4f}")
    print(f"  max  : {df['SCI_t'].max():.4f}")
    print(f"  min  : {df['SCI_t'].min():.4f}")

    print(f"\nΔM_t stats:")
    print(f"  mean : {df['delta_M_t'].mean():.4f}")
    print(f"  std  : {df['delta_M_t'].std():.4f}")
    print(f"  max  : {df['delta_M_t'].max():.4f}")

    print(f"\nSample output:")
    print(df[["date", "delta_M_t", "HR_t", "SCI_t"]].head(10).to_string(index=False))

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"sci_timeseries_{mode}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    return df


def compute_both_sci(**kwargs):
    """
    Compute SCI_t for both baselines.
    Returns (df_benign, df_adv).
    """
    df_benign = compute_sci_timeseries("benign", **kwargs)
    df_adv    = compute_sci_timeseries("adv",    **kwargs)
    return df_benign, df_adv


if __name__ == "__main__":
    compute_both_sci()