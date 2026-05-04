"""
compute_sci_combined_drift.py
------------------------------
Runs the SCI pipeline on combined-drift baselines.
All helpers inlined — no project imports needed.

Expected result:
    SCI reacts STRONGER than structure-only alone
    -> combined drift amplifies structural coordination signal
    -> feature changes reinforce structural patterns

Reads:
    graphs/baselines/*_combined_drift_{intensity}.gpickle
    graphs/sci_results/sci_timeseries_benign.csv  (for comparison)

Outputs:
    graphs/sci_results/sci_timeseries_combined_drift_low.csv
    graphs/sci_results/sci_timeseries_combined_drift_medium.csv
    graphs/sci_results/sci_timeseries_combined_drift_high.csv
    graphs/sci_results/sci_combined_drift_cv_comparison.csv
    graphs/sci_results/sci_combined_drift_overlay.png
    graphs/sci_results/sci_ablation_summary.png   <- all 3 experiments together
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

COLORS_COMBINED = {
    "benign":               "#5B9BD5",
    "combined_drift_low":   "#C084C8",
    "combined_drift_medium":"#9B4ECA",
    "combined_drift_high":  "#6A0DAD",
}

COLORS_ABLATION = {
    "benign":               "#5B9BD5",
    "struct_drift_high":    "#E06C75",
    "feature_drift_high":   "#F0A500",
    "combined_drift_high":  "#6A0DAD",
}


# ── SCI helpers (inlined) ─────────────────────────────────────────────────

def count_motifs(G_snap):
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
                shared = (tx_to_entities.get(txs[i], set()) &
                          tx_to_entities.get(txs[j], set()))
                if len(shared) > 1:
                    triangles += 1

    return {"triangles": triangles // 3, "wedges": wedges}


def compute_degree_entropy(G_snap):
    degrees = [d for _, d in G_snap.degree()]
    total   = sum(degrees)
    if total == 0:
        return 0.0
    probs = [d / total for d in degrees if d > 0]
    return float(-sum(p * np.log(p + 1e-12) for p in probs))


def compute_top_k_centrality(G_snap, k=20):
    deg_centrality = nx.degree_centrality(G_snap)
    top_k = sorted(deg_centrality.values(), reverse=True)[:k]
    return np.array(top_k, dtype=np.float32)


def zscore_col(series):
    mu, std = series.mean(), series.std()
    return series * 0.0 if std < 1e-12 else (series - mu) / std


# ── per-intensity SCI ─────────────────────────────────────────────────────

def compute_sci_for_combined_drift(intensity):
    mode    = f"combined_drift_{intensity}"
    pattern = os.path.join(BASELINE_DIR, f"*_{mode}.gpickle")
    paths   = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No {mode} baselines in {BASELINE_DIR}\n"
            "Run create_combined_drift.py first."
        )

    print(f"\n  [{mode}] -- {len(paths)} snapshots")

    records = []
    prev_tri = prev_wed = prev_ent = prev_cent = None

    for path in paths:
        date_str = os.path.basename(path).replace(f"_{mode}.gpickle", "")

        with open(path, "rb") as f:
            G_snap = pickle.load(f)

        motifs  = count_motifs(G_snap)
        entropy = compute_degree_entropy(G_snap)
        cent    = compute_top_k_centrality(G_snap)

        tri = motifs["triangles"]
        wed = motifs["wedges"]

        if prev_tri is None:
            delta_tri = delta_wed = delta_ent = delta_cent = float("nan")
        else:
            delta_tri  = float(abs(tri - prev_tri))
            delta_wed  = float(abs(wed - prev_wed))
            delta_ent  = float(abs(entropy - prev_ent))
            min_k      = min(len(prev_cent), len(cent))
            delta_cent = float(np.linalg.norm(cent[:min_k] - prev_cent[:min_k]))

        records.append({
            "date": date_str,
            "triangles": tri, "wedges": wed, "entropy": entropy,
            "delta_tri": delta_tri, "delta_wed": delta_wed,
            "delta_ent": delta_ent, "delta_cent": delta_cent,
        })
        prev_tri, prev_wed, prev_ent, prev_cent = tri, wed, entropy, cent

    df    = pd.DataFrame(records)
    valid = df["delta_tri"].notna()

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

    df.loc[valid, "SCI_t"] = (
        0.5 * zscore_col(df.loc[valid, "delta_M_t"]) +
        0.5 * zscore_col(df.loc[valid, "delta_cent"])
    ).values

    return df


# ── combined drift overlay plot ───────────────────────────────────────────

def plot_combined_drift_overlay(all_dfs):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)
    fig.suptitle(
        "SCI -- Combined Drift vs Benign Baseline\n"
        "(Both structure + feature changed simultaneously)",
        fontsize=13, fontweight="bold"
    )

    signals = [
        ("delta_M_t", "ΔM_t (Motif Signal)"),
        ("HR_t",      "HR_t (Hub Signal)"),
        ("SCI_t",     "SCI_t (Unified)"),
    ]

    for ax, (col, label) in zip(axes, signals):
        for mode, df in all_dfs.items():
            valid = df.dropna(subset=[col])
            if valid.empty:
                continue
            series   = valid[col]
            x        = np.arange(len(series))
            color    = COLORS_COMBINED.get(mode, "gray")
            lw       = 2.0 if mode == "benign" else 1.2
            mu       = series.mean()
            std      = series.std()
            cv       = std / mu if mu > 0 else float("nan")
            cv_label = f"CV={cv:.3f}" if not np.isnan(cv) else "CV=n/a"
            ax.plot(x, series.values, color=color, linewidth=lw,
                    alpha=0.85, label=f"{mode} ({cv_label})")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(SCI_DIR, "sci_combined_drift_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {out}")
    plt.close()


# ── ablation summary plot ─────────────────────────────────────────────────

def plot_sci_ablation_summary(sci_dir):
    """
    3-row figure: benign vs struct_high vs feature_high vs combined_high
    Main paper figure for SCI Coupling & Independence section.
    """
    files = {
        "benign":               "sci_timeseries_benign.csv",
        "struct_drift_high":    "sci_timeseries_struct_drift_high.csv",
        "feature_drift_high":   "sci_timeseries_feature_drift_high.csv",
        "combined_drift_high":  "sci_timeseries_combined_drift_high.csv",
    }

    dfs = {}
    for mode, fname in files.items():
        path = os.path.join(sci_dir, fname)
        if os.path.exists(path):
            dfs[mode] = pd.read_csv(path)
        else:
            print(f"  Warning: {path} not found — skipping")

    if len(dfs) < 2:
        print("  Not enough data for SCI ablation summary — skipping")
        return

    signals = [
        ("delta_M_t", "ΔM_t (Motif Signal)"),
        ("HR_t",      "HR_t (Hub Signal)"),
        ("SCI_t",     "SCI_t (Unified)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)
    fig.suptitle(
        "SCI Ablation Summary — Benign vs Structure / Feature / Combined Drift\n"
        "(High intensity only — main Coupling & Independence result)",
        fontsize=12, fontweight="bold"
    )

    for ax, (col, label) in zip(axes, signals):
        for mode, df in dfs.items():
            valid = df.dropna(subset=[col])
            if valid.empty:
                continue
            series   = valid[col]
            x        = np.arange(len(series))
            color    = COLORS_ABLATION.get(mode, "gray")
            lw       = 2.0 if mode == "benign" else 1.4
            mu       = series.mean()
            std      = series.std()
            cv       = std / mu if mu > 0 else float("nan")
            cv_label = f"CV={cv:.3f}" if not np.isnan(cv) else "CV=n/a"
            ax.plot(x, series.values, color=color, linewidth=lw,
                    alpha=0.9, label=f"{mode} ({cv_label})")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(sci_dir, "sci_ablation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────

def run_sci_combined_drift():
    print("\n" + "=" * 65)
    print("  SCI -- COMBINED DRIFT ANALYSIS")
    print("  Expected: stronger reaction than structure-only alone")
    print("=" * 65)

    all_dfs = {}

    benign_path = os.path.join(SCI_DIR, "sci_timeseries_benign.csv")
    if os.path.exists(benign_path):
        all_dfs["benign"] = pd.read_csv(benign_path)
        print(f"  Loaded benign SCI: {benign_path}")
    else:
        print(f"  Warning: benign SCI not found at {benign_path}")

    for intensity in INTENSITIES:
        mode = f"combined_drift_{intensity}"
        df   = compute_sci_for_combined_drift(intensity)
        out  = os.path.join(SCI_DIR, f"sci_timeseries_{mode}.csv")
        df.to_csv(out, index=False)
        print(f"  Saved -> {out}")
        all_dfs[mode] = df

    # CV table
    rows = []
    for sig in ["delta_M_t", "HR_t", "SCI_t"]:
        row = {"signal": sig}
        for mode, df in all_dfs.items():
            s         = df[sig].dropna()
            row[mode] = round(s.std() / s.mean(), 4) if s.mean() > 0 else float("nan")
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(os.path.join(SCI_DIR, "sci_combined_drift_cv_comparison.csv"),
                 index=False)

    print("\n" + "=" * 65)
    print("  CV COMPARISON -- SCI Combined Drift")
    print(f"\n{cv_df.to_string(index=False)}")
    print("=" * 65)

    plot_combined_drift_overlay(all_dfs)

    print("\n  Generating SCI ablation summary plot...")
    plot_sci_ablation_summary(SCI_DIR)

    print("\n" + "=" * 65)
    print("  DONE -- Coupling & Independence Analysis COMPLETE")
    print("  All three ablations finished:")
    print("    structure-only -> feature-only -> combined")
    print(f"  Outputs -> {SCI_DIR}")
    print("=" * 65)
    return all_dfs


if __name__ == "__main__":
    run_sci_combined_drift()