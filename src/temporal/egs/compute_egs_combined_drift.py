"""
compute_egs_combined_drift.py
------------------------------
Computes EGS sub-signals (ΔP, ΔD, ΔA) and unified EGS_t for all three
combined-drift intensity levels. All helpers inlined — no project imports.

Expected result:
    EGS reacts STRONGER than structure-only or feature-only alone
    -> combined manipulation amplifies embedding instability

Outputs:
    graphs/egs_results/egs_timeseries_combined_drift_low.csv
    graphs/egs_results/egs_timeseries_combined_drift_medium.csv
    graphs/egs_results/egs_timeseries_combined_drift_high.csv
    graphs/egs_results/egs_combined_drift_cv_comparison.csv
    graphs/egs_results/egs_combined_drift_overlay.png
    graphs/egs_results/egs_ablation_summary.png   <- all 3 experiments together
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

EMBED_DIR = os.path.join(_PROJECT_ROOT, "graphs", "embeddings")
EGS_DIR   = os.path.join(_PROJECT_ROOT, "graphs", "egs_results")
os.makedirs(EGS_DIR, exist_ok=True)

INTENSITIES = ["low", "medium", "high"]

# colors for combined drift
COLORS_COMBINED = {
    "benign":               "#5B9BD5",
    "combined_drift_low":   "#C084C8",
    "combined_drift_medium":"#9B4ECA",
    "combined_drift_high":  "#6A0DAD",
}

# colors for full ablation summary plot
COLORS_ABLATION = {
    "benign":                "#5B9BD5",
    "struct_drift_high":     "#E06C75",
    "feature_drift_high":    "#F0A500",
    "combined_drift_high":   "#6A0DAD",
}


# ── geometry helpers (inlined) ────────────────────────────────────────────

def compute_prototype(emb):
    return emb.mean(axis=0)

def compute_dispersion(emb, prototype):
    return float(np.linalg.norm(emb - prototype, axis=1).mean())

def compute_anisotropy(emb, prototype):
    n, d = emb.shape
    if n < 2:
        return 1.0 / d
    centered = emb - prototype
    if n >= d:
        eigenvalues = np.linalg.eigvalsh(np.cov(centered.T))
    else:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s ** 2) / (n - 1)
    eigenvalues = np.abs(eigenvalues)
    total = eigenvalues.sum()
    return 1.0 / d if total < 1e-12 else float(eigenvalues.max() / total)

def zscore(series):
    mu, std = series.mean(), series.std()
    return series * 0.0 if std < 1e-12 else (series - mu) / std


# ── per-intensity EGS ─────────────────────────────────────────────────────

def compute_egs_for_combined_drift(intensity):
    mode      = f"combined_drift_{intensity}"
    embed_dir = os.path.join(EMBED_DIR, mode)
    npy_files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files in:\n  {embed_dir}\n"
            "Run embed_snapshots_combined_drift.py first."
        )

    print(f"\n  [{mode}] -- {len(npy_files)} daily files")

    records                             = []
    prev_proto = prev_disp = prev_aniso = None

    for npy_path in npy_files:
        date_str  = os.path.basename(npy_path).replace(".npy", "")
        meta_path = os.path.join(embed_dir, f"meta_{date_str}.pkl")

        emb = np.load(npy_path)
        with open(meta_path, "rb") as f:
            pickle.load(f)   # load meta (unused but keeps pattern consistent)

        proto = compute_prototype(emb)
        disp  = compute_dispersion(emb, proto)
        aniso = compute_anisotropy(emb, proto)

        if prev_proto is None:
            delta_P = delta_D = delta_A = float("nan")
        else:
            delta_P = float(np.linalg.norm(proto - prev_proto))
            delta_D = float(abs(disp  - prev_disp))
            delta_A = float(abs(aniso - prev_aniso))

        records.append({
            "date": date_str, "n_tx": emb.shape[0],
            "dispersion": disp, "anisotropy": aniso,
            "delta_P": delta_P, "delta_D": delta_D, "delta_A": delta_A,
        })
        prev_proto, prev_disp, prev_aniso = proto, disp, aniso

    return records


def add_unified_egs(df):
    valid = df["delta_P"].notna()
    df    = df.copy()
    for col in ["delta_P", "delta_D", "delta_A"]:
        df[f"{col}_z"] = float("nan")
        df.loc[valid, f"{col}_z"] = zscore(df.loc[valid, col]).values
    df["EGS_t"] = float("nan")
    df.loc[valid, "EGS_t"] = (
        (1/3) * df.loc[valid, "delta_P_z"] +
        (1/3) * df.loc[valid, "delta_D_z"] +
        (1/3) * df.loc[valid, "delta_A_z"]
    )
    return df


# ── combined drift overlay plot ───────────────────────────────────────────

def plot_combined_drift_overlay(all_dfs):
    signals = [
        ("delta_P", "ΔP_t (Prototype Movement)"),
        ("delta_D", "ΔD_t (Dispersion Change)"),
        ("delta_A", "ΔA_t (Anisotropy Change)"),
        ("EGS_t",   "EGS_t (Unified)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=False)
    fig.suptitle(
        "Combined Drift — EGS Signals vs Benign Baseline\n"
        "(Both structure + feature changed simultaneously)",
        fontsize=13, fontweight="bold"
    )

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

    out = os.path.join(EGS_DIR, "egs_combined_drift_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {out}")
    plt.close()


# ── ablation summary plot (all 3 experiments at high intensity) ───────────

def plot_ablation_summary(egs_dir):
    """
    4-row figure showing benign vs struct_drift_high vs feature_drift_high
    vs combined_drift_high for each EGS signal.
    This is the main paper figure for Coupling & Independence Analysis.
    """
    # load all needed CSVs
    files = {
        "benign":               "unified_egs_benign.csv",
        "struct_drift_high":    "egs_timeseries_struct_drift_high.csv",
        "feature_drift_high":   "egs_timeseries_feature_drift_high.csv",
        "combined_drift_high":  "egs_timeseries_combined_drift_high.csv",
    }

    dfs = {}
    for mode, fname in files.items():
        path = os.path.join(egs_dir, fname)
        if os.path.exists(path):
            dfs[mode] = pd.read_csv(path)
        else:
            print(f"  Warning: {path} not found — skipping from ablation summary")

    if len(dfs) < 2:
        print("  Not enough data for ablation summary — skipping")
        return

    signals = [
        ("delta_P", "ΔP_t (Prototype Movement)"),
        ("delta_D", "ΔD_t (Dispersion Change)"),
        ("delta_A", "ΔA_t (Anisotropy Change)"),
        ("EGS_t",   "EGS_t (Unified)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=False)
    fig.suptitle(
        "EGS Ablation Summary — Benign vs Structure / Feature / Combined Drift\n"
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

    out = os.path.join(egs_dir, "egs_ablation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {out}")
    plt.close()


# ── CV table ──────────────────────────────────────────────────────────────

def build_cv_table(all_dfs):
    rows = []
    for sig in ["delta_P", "delta_D", "delta_A", "EGS_t"]:
        row = {"signal": sig}
        for mode, df in all_dfs.items():
            s         = df[sig].dropna()
            row[mode] = round(s.std() / s.mean(), 4) if s.mean() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────

def run_egs_combined_drift():
    print("\n" + "=" * 65)
    print("  EGS -- COMBINED DRIFT ANALYSIS")
    print("  Expected: strongest reaction across all signals")
    print("=" * 65)

    all_dfs = {}

    benign_path = os.path.join(EGS_DIR, "unified_egs_benign.csv")
    if os.path.exists(benign_path):
        all_dfs["benign"] = pd.read_csv(benign_path)
        print(f"  Loaded benign baseline: {benign_path}")
    else:
        print(f"  Warning: benign baseline not found at {benign_path}")

    for intensity in INTENSITIES:
        mode    = f"combined_drift_{intensity}"
        records = compute_egs_for_combined_drift(intensity)
        df      = add_unified_egs(pd.DataFrame(records))
        out     = os.path.join(EGS_DIR, f"egs_timeseries_{mode}.csv")
        df.to_csv(out, index=False)
        print(f"  Saved -> {out}")
        all_dfs[mode] = df

    cv_df = build_cv_table(all_dfs)
    cv_df.to_csv(os.path.join(EGS_DIR, "egs_combined_drift_cv_comparison.csv"),
                 index=False)

    print("\n" + "=" * 65)
    print("  CV COMPARISON -- Benign vs Combined Drift")
    print(f"\n{cv_df.to_string(index=False)}")
    print("=" * 65)

    plot_combined_drift_overlay(all_dfs)

    # ablation summary — requires all 3 experiments to have run
    print("\n  Generating ablation summary plot...")
    plot_ablation_summary(EGS_DIR)

    print("\n" + "=" * 65)
    print("  DONE -- Next: run compute_sci_combined_drift.py")
    print(f"  Outputs -> {EGS_DIR}")
    print("=" * 65)
    return all_dfs


if __name__ == "__main__":
    run_egs_combined_drift()