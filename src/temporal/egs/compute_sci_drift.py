"""
compute_sci_drift.py
---------------------
Runs the SCI pipeline on structure-drift baselines and compares
against the original benign SCI baseline.

Key expected result:
    Benign SCI_t   → flat, low CV  (no structural changes)
    Drift SCI_t    → higher CV, more spikes (edges rewired → structure changed)
    This PROVES SCI is sensitive to structural changes.

Reads:
    graphs/baselines/*_struct_drift_{intensity}.gpickle
    graphs/sci_results/sci_timeseries_benign.csv   (for comparison)

Outputs:
    graphs/sci_results/sci_timeseries_struct_drift_low.csv
    graphs/sci_results/sci_timeseries_struct_drift_medium.csv
    graphs/sci_results/sci_timeseries_struct_drift_high.csv
    graphs/sci_results/sci_drift_cv_comparison.csv
    graphs/sci_results/sci_drift_overlay.png
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# ── paths ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR = os.path.join(_PROJECT_ROOT, "graphs", "baselines")
SCI_DIR      = os.path.join(_PROJECT_ROOT, "graphs", "sci_results")
os.makedirs(SCI_DIR, exist_ok=True)

INTENSITIES = ["low", "medium", "high"]
MAX_FAN_OUT = 500

COLORS = {
    "benign":              "#5B9BD5",
    "struct_drift_low":    "#98C99E",
    "struct_drift_medium": "#F0A500",
    "struct_drift_high":   "#E06C75",
}


# ── motif counting (same logic as your existing SCI pipeline) ─────────────

def count_motifs(G_snap):
    """Count triangles, wedges, fraud-legit-fraud chains in snapshot."""
    entity_to_txs  = {}
    tx_to_entities = {}

    for node, data in G_snap.nodes(data=True):
        if data.get("node_type") != "transaction":
            txs = [nb for nb in G_snap.neighbors(node)
                   if G_snap.nodes[nb].get("node_type") == "transaction"]
            entity_to_txs[node] = txs
        else:
            ents = [nb for nb in G_snap.neighbors(node)
                    if G_snap.nodes[nb].get("node_type") != "transaction"]
            tx_to_entities[node] = set(ents)

    triangles = 0
    wedges    = 0

    for entity, txs in entity_to_txs.items():
        k = len(txs)
        if k > MAX_FAN_OUT:
            continue
        wedges += k * (k - 1) // 2
        for i in range(k):
            for j in range(i + 1, k):
                shared = tx_to_entities.get(txs[i], set()) & \
                         tx_to_entities.get(txs[j], set())
                if len(shared) > 1:   # share another entity besides current
                    triangles += 1

    triangles //= 3   # each triangle counted 3x

    return {"triangles": triangles, "wedges": wedges}


def compute_degree_entropy(G_snap):
    degrees = [d for _, d in G_snap.degree()]
    total   = sum(degrees)
    if total == 0:
        return 0.0
    probs   = [d / total for d in degrees if d > 0]
    return float(-sum(p * np.log(p + 1e-12) for p in probs))


def compute_top_k_centrality(G_snap, k=20):
    deg_centrality = nx.degree_centrality(G_snap)
    top_k = sorted(deg_centrality.values(), reverse=True)[:k]
    return np.array(top_k, dtype=np.float32)


# ── per-day SCI ───────────────────────────────────────────────────────────

def compute_sci_timeseries(intensity: str) -> pd.DataFrame:
    mode    = f"struct_drift_{intensity}"
    pattern = os.path.join(BASELINE_DIR, f"*_{mode}.gpickle")
    paths   = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No {mode} baselines in {BASELINE_DIR}\n"
            f"Run create_structure_drift.py first."
        )

    print(f"\n  [{mode}] — {len(paths)} snapshots")

    records         = []
    prev_triangles  = None
    prev_wedges     = None
    prev_entropy    = None
    prev_centrality = None

    for path in paths:
        date_str = os.path.basename(path).replace(f"_{mode}.gpickle", "")

        with open(path, "rb") as f:
            G_snap = pickle.load(f)

        motifs  = count_motifs(G_snap)
        entropy = compute_degree_entropy(G_snap)
        cent    = compute_top_k_centrality(G_snap)

        tri = motifs["triangles"]
        wed = motifs["wedges"]

        if prev_triangles is None:
            delta_tri  = float("nan")
            delta_wed  = float("nan")
            delta_ent  = float("nan")
            delta_cent = float("nan")
        else:
            delta_tri  = float(abs(tri - prev_triangles))
            delta_wed  = float(abs(wed - prev_wedges))
            delta_ent  = float(abs(entropy - prev_entropy))
            # pad/trim to same length
            prev_c = prev_centrality
            curr_c = cent
            min_k  = min(len(prev_c), len(curr_c))
            delta_cent = float(np.linalg.norm(curr_c[:min_k] - prev_c[:min_k]))

        records.append({
            "date":       date_str,
            "triangles":  tri,
            "wedges":     wed,
            "entropy":    entropy,
            "delta_tri":  delta_tri,
            "delta_wed":  delta_wed,
            "delta_ent":  delta_ent,
            "delta_cent": delta_cent,
        })

        prev_triangles  = tri
        prev_wedges     = wed
        prev_entropy    = entropy
        prev_centrality = cent

    df = pd.DataFrame(records)

    # build ΔM_t (motif unified) and HR_t (hub unified), then SCI_t
    valid = df["delta_tri"].notna()

    def zscore_col(series):
        mu, std = series.mean(), series.std()
        return (series - mu) / std if std > 1e-12 else series * 0.0

    df["delta_M_t"] = float("nan")
    df["HR_t"]      = float("nan")
    df["SCI_t"]     = float("nan")

    df.loc[valid, "delta_M_t"] = (
        0.5 * zscore_col(df.loc[valid, "delta_tri"]) +
        0.5 * zscore_col(df.loc[valid, "delta_wed"])
    ).values

    df.loc[valid, "HR_t"] = (
        0.5 * zscore_col(df.loc[valid, "delta_ent"]) +
        0.5 * zscore_col(df.loc[valid, "delta_cent"])
    ).values

    # re-zscore before combining
    dm_z  = zscore_col(df.loc[valid, "delta_M_t"])
    hr_z  = zscore_col(df.loc[valid, "HR_t"])
    df.loc[valid, "SCI_t"] = (0.5 * dm_z + 0.5 * hr_z).values

    return df


# ── overlay plot ──────────────────────────────────────────────────────────

def plot_sci_drift_overlay(all_dfs: dict):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)
    fig.suptitle(
        "SCI — Structure-Only Drift vs Benign Baseline\n"
        "(Proves SCI reacts to structural changes)",
        fontsize=13, fontweight="bold"
    )

    signals = [
        ("delta_M_t", "ΔM_t (Motif Signal)"),
        ("HR_t",      "HR_t (Hub Signal)"),
        ("SCI_t",     "SCI_t (Unified)"),
    ]

    for ax, (col, label) in zip(axes, signals):
        for mode, df in all_dfs.items():
            valid  = df.dropna(subset=[col])
            series = valid[col]
            x      = np.arange(len(series))
            color  = COLORS.get(mode, "gray")
            lw     = 2.0 if mode == "benign" else 1.2

            mu  = series.mean()
            std = series.std()
            cv  = std / mu if mu > 0 else float("nan")
            cv_label = f"CV={cv:.3f}" if not np.isnan(cv) else "CV=n/a"

            ax.plot(x, series.values, color=color, linewidth=lw,
                    alpha=0.85, label=f"{mode} ({cv_label})")

        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(SCI_DIR, "sci_drift_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {out}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────

def run_sci_drift():
    print("\n" + "=" * 65)
    print("  SCI — STRUCTURE-ONLY DRIFT ANALYSIS")
    print("  Expected: drift SCI_t > benign SCI_t (proves structural sensitivity)")
    print("=" * 65)

    all_dfs = {}

    # load existing benign SCI for comparison
    benign_sci_path = os.path.join(SCI_DIR, "sci_timeseries_benign.csv")
    if os.path.exists(benign_sci_path):
        all_dfs["benign"] = pd.read_csv(benign_sci_path)
        print(f"  Loaded benign SCI baseline: {benign_sci_path}")
    else:
        print(f"  ⚠ Benign SCI not found at {benign_sci_path} — skipping comparison")

    for intensity in INTENSITIES:
        mode = f"struct_drift_{intensity}"
        df   = compute_sci_timeseries(intensity)

        out_path = os.path.join(SCI_DIR, f"sci_timeseries_{mode}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        all_dfs[mode] = df

    # CV table
    rows = []
    for signal in ["delta_M_t", "HR_t", "SCI_t"]:
        row = {"signal": signal}
        for mode, df in all_dfs.items():
            s  = df[signal].dropna()
            cv = s.std() / s.mean() if s.mean() > 0 else float("nan")
            row[mode] = round(cv, 4)
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    cv_path = os.path.join(SCI_DIR, "sci_drift_cv_comparison.csv")
    cv_df.to_csv(cv_path, index=False)

    print("\n" + "=" * 65)
    print("  CV COMPARISON — SCI Signals")
    print(f"  {cv_df.to_string(index=False)}")
    print("=" * 65)

    plot_sci_drift_overlay(all_dfs)

    print("\n" + "=" * 65)
    print("  DONE — Structure-Only SCI Drift Analysis Complete")
    print(f"  All outputs → {SCI_DIR}")
    print("=" * 65)

    return all_dfs


if __name__ == "__main__":
    run_sci_drift()