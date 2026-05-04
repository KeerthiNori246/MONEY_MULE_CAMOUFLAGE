"""
compute_egs_drift.py
---------------------
Computes EGS sub-signals (ΔP, ΔD, ΔA) and unified EGS_t for all three
structure-drift intensity levels.

NOTE: All helper functions are inlined here — no imports from other
project files needed. Just standard library + numpy/pandas/matplotlib.

Outputs:
    graphs/egs_results/egs_timeseries_struct_drift_low.csv
    graphs/egs_results/egs_timeseries_struct_drift_medium.csv
    graphs/egs_results/egs_timeseries_struct_drift_high.csv
    graphs/egs_results/egs_drift_cv_comparison.csv
    graphs/egs_results/egs_drift_overlay.png
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths (relative to this file's location) ─────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

EMBED_DIR = os.path.join(_PROJECT_ROOT, "graphs", "embeddings")
EGS_DIR   = os.path.join(_PROJECT_ROOT, "graphs", "egs_results")
os.makedirs(EGS_DIR, exist_ok=True)

INTENSITIES = ["low", "medium", "high"]

COLORS = {
    "benign":              "#5B9BD5",
    "struct_drift_low":    "#98C99E",
    "struct_drift_medium": "#F0A500",
    "struct_drift_high":   "#E06C75",
}


# ── geometry helpers (inlined — no external project imports needed) ───────

def compute_prototype(emb: np.ndarray) -> np.ndarray:
    return emb.mean(axis=0)


def compute_dispersion(emb: np.ndarray, prototype: np.ndarray) -> float:
    diffs = emb - prototype
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.mean())


def compute_anisotropy(emb: np.ndarray, prototype: np.ndarray) -> float:
    n, d = emb.shape
    if n < 2:
        return 1.0 / d
    centered = emb - prototype
    if n >= d:
        cov         = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
    else:
        _, s, _     = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s ** 2) / (n - 1)
    eigenvalues = np.abs(eigenvalues)
    total = eigenvalues.sum()
    if total < 1e-12:
        return 1.0 / d
    return float(eigenvalues.max() / total)


def zscore(series: pd.Series) -> pd.Series:
    mu, std = series.mean(), series.std()
    if std < 1e-12:
        return series * 0.0
    return (series - mu) / std


# ── per-intensity EGS computation ────────────────────────────────────────

def compute_egs_for_drift(intensity: str) -> list:
    mode      = f"struct_drift_{intensity}"
    embed_dir = os.path.join(EMBED_DIR, mode)
    npy_files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in:\n  {embed_dir}\n"
            f"Run embed_snapshots_drift.py first."
        )

    print(f"\n  [{mode}] — {len(npy_files)} daily files")

    records    = []
    prev_proto = None
    prev_disp  = None
    prev_aniso = None

    for npy_path in npy_files:
        date_str  = os.path.basename(npy_path).replace(".npy", "")
        meta_path = os.path.join(embed_dir, f"meta_{date_str}.pkl")

        emb = np.load(npy_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        n_tx  = emb.shape[0]
        proto = compute_prototype(emb)
        disp  = compute_dispersion(emb, proto)
        aniso = compute_anisotropy(emb, proto)

        if prev_proto is None:
            delta_P = float("nan")
            delta_D = float("nan")
            delta_A = float("nan")
        else:
            delta_P = float(np.linalg.norm(proto - prev_proto))
            delta_D = float(abs(disp  - prev_disp))
            delta_A = float(abs(aniso - prev_aniso))

        records.append({
            "date":       date_str,
            "n_tx":       n_tx,
            "dispersion": disp,
            "anisotropy": aniso,
            "delta_P":    delta_P,
            "delta_D":    delta_D,
            "delta_A":    delta_A,
        })

        prev_proto = proto
        prev_disp  = disp
        prev_aniso = aniso

    return records


def add_unified_egs(df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = df["delta_P"].notna()
    df = df.copy()
    for col in ["delta_P", "delta_D", "delta_A"]:
        z_col = f"{col}_z"
        df[z_col] = float("nan")
        df.loc[valid_mask, z_col] = zscore(df.loc[valid_mask, col]).values
    df["EGS_t"] = float("nan")
    df.loc[valid_mask, "EGS_t"] = (
        (1/3) * df.loc[valid_mask, "delta_P_z"] +
        (1/3) * df.loc[valid_mask, "delta_D_z"] +
        (1/3) * df.loc[valid_mask, "delta_A_z"]
    )
    return df


# ── overlay plot ──────────────────────────────────────────────────────────

def plot_drift_overlay(all_dfs: dict):
    signals = [
        ("delta_P", "ΔP_t (Prototype Movement)"),
        ("delta_D", "ΔD_t (Dispersion Change)"),
        ("delta_A", "ΔA_t (Anisotropy Change)"),
        ("EGS_t",   "EGS_t (Unified)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=False)
    fig.suptitle(
        "Structure-Only Drift — EGS Signals vs Benign Baseline",
        fontsize=13, fontweight="bold"
    )

    for ax, (col, label) in zip(axes, signals):
        for mode, df in all_dfs.items():
            valid = df.dropna(subset=[col])
            if valid.empty:
                continue
            series = valid[col]
            x      = np.arange(len(series))
            color  = COLORS.get(mode, "gray")
            lw     = 2.0 if mode == "benign" else 1.2
            mu     = series.mean()
            std    = series.std()
            cv     = std / mu if mu > 0 else float("nan")
            cv_label = f"CV={cv:.3f}" if not np.isnan(cv) else "CV=n/a"
            ax.plot(x, series.values, color=color, linewidth=lw,
                    alpha=0.85, label=f"{mode} ({cv_label})")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(EGS_DIR, "egs_drift_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {out}")
    plt.close()


# ── CV comparison table ───────────────────────────────────────────────────

def build_cv_table(all_dfs: dict) -> pd.DataFrame:
    rows = []
    for signal in ["delta_P", "delta_D", "delta_A", "EGS_t"]:
        row = {"signal": signal}
        for mode, df in all_dfs.items():
            s  = df[signal].dropna()
            cv = s.std() / s.mean() if s.mean() > 0 else float("nan")
            row[mode] = round(cv, 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────

def run_egs_drift():
    print("\n" + "=" * 65)
    print("  EGS — STRUCTURE-ONLY DRIFT ANALYSIS")
    print("=" * 65)

    all_dfs = {}

    # load benign baseline for comparison
    benign_path = os.path.join(EGS_DIR, "unified_egs_benign.csv")
    if os.path.exists(benign_path):
        all_dfs["benign"] = pd.read_csv(benign_path)
        print(f"  Loaded benign baseline: {benign_path}")
    else:
        print(f"  ⚠  Benign unified EGS not found at:\n     {benign_path}")
        print(f"     Run compute_unified_egs.py first for comparison.")
        print(f"     Continuing without benign overlay...\n")

    # compute for each drift intensity
    for intensity in INTENSITIES:
        mode    = f"struct_drift_{intensity}"
        records = compute_egs_for_drift(intensity)
        df      = pd.DataFrame(records)
        df      = add_unified_egs(df)

        out_path = os.path.join(EGS_DIR, f"egs_timeseries_{mode}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        all_dfs[mode] = df

    # CV comparison
    cv_df   = build_cv_table(all_dfs)
    cv_path = os.path.join(EGS_DIR, "egs_drift_cv_comparison.csv")
    cv_df.to_csv(cv_path, index=False)

    print("\n" + "=" * 65)
    print("  CV COMPARISON — Benign vs Structure Drift")
    print(f"\n{cv_df.to_string(index=False)}")
    print("=" * 65)

    plot_drift_overlay(all_dfs)

    print("\n" + "=" * 65)
    print("  DONE — Next: run compute_sci_drift.py")
    print(f"  Outputs → {EGS_DIR}")
    print("=" * 65)

    return all_dfs


if __name__ == "__main__":
    run_egs_drift()