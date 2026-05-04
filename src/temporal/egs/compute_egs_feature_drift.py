"""
compute_egs_feature_drift.py
-----------------------------
Computes EGS sub-signals (ΔP, ΔD, ΔA) and unified EGS_t for all three
feature-drift intensity levels. All helpers inlined — no project imports.

Expected result:
    EGS reacts   -> embeddings are feature-sensitive (proves EGS depends on features)
    SCI stays flat -> confirmed in compute_sci_feature_drift.py

Outputs:
    graphs/egs_results/egs_timeseries_feature_drift_low.csv
    graphs/egs_results/egs_timeseries_feature_drift_medium.csv
    graphs/egs_results/egs_timeseries_feature_drift_high.csv
    graphs/egs_results/egs_feature_drift_cv_comparison.csv
    graphs/egs_results/egs_feature_drift_overlay.png
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

COLORS = {
    "benign":                "#5B9BD5",
    "feature_drift_low":    "#98C99E",
    "feature_drift_medium": "#F0A500",
    "feature_drift_high":   "#E06C75",
}


# ── geometry helpers (inlined) ────────────────────────────────────────────

def compute_prototype(emb: np.ndarray) -> np.ndarray:
    return emb.mean(axis=0)


def compute_dispersion(emb: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.linalg.norm(emb - prototype, axis=1).mean())


def compute_anisotropy(emb: np.ndarray, prototype: np.ndarray) -> float:
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


def zscore(series: pd.Series) -> pd.Series:
    mu, std = series.mean(), series.std()
    return series * 0.0 if std < 1e-12 else (series - mu) / std


# ── per-intensity EGS ─────────────────────────────────────────────────────

def compute_egs_for_feature_drift(intensity: str) -> list:
    mode      = f"feature_drift_{intensity}"
    embed_dir = os.path.join(EMBED_DIR, mode)
    npy_files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files in:\n  {embed_dir}\n"
            "Run embed_snapshots_feature_drift.py first."
        )

    print(f"\n  [{mode}] -- {len(npy_files)} daily files")

    records                             = []
    prev_proto = prev_disp = prev_aniso = None

    for npy_path in npy_files:
        date_str  = os.path.basename(npy_path).replace(".npy", "")
        meta_path = os.path.join(embed_dir, f"meta_{date_str}.pkl")

        emb = np.load(npy_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

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


def add_unified_egs(df: pd.DataFrame) -> pd.DataFrame:
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


# ── overlay plot ──────────────────────────────────────────────────────────

def plot_feature_drift_overlay(all_dfs: dict):
    signals = [
        ("delta_P", "ΔP_t (Prototype Movement)"),
        ("delta_D", "ΔD_t (Dispersion Change)"),
        ("delta_A", "ΔA_t (Anisotropy Change)"),
        ("EGS_t",   "EGS_t (Unified)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=False)
    fig.suptitle(
        "Feature-Only Drift — EGS Signals vs Benign Baseline\n"
        "(EGS should react — proves embeddings depend on features)",
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
            lbl    = f"CV={cv:.3f}" if not np.isnan(cv) else "CV=n/a"
            ax.plot(x, series.values, color=color, linewidth=lw,
                    alpha=0.85, label=f"{mode} ({lbl})")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(EGS_DIR, "egs_feature_drift_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {out}")
    plt.close()


# ── CV table ──────────────────────────────────────────────────────────────

def build_cv_table(all_dfs: dict) -> pd.DataFrame:
    rows = []
    for sig in ["delta_P", "delta_D", "delta_A", "EGS_t"]:
        row = {"signal": sig}
        for mode, df in all_dfs.items():
            s       = df[sig].dropna()
            row[mode] = round(s.std() / s.mean(), 4) if s.mean() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────

def run_egs_feature_drift():
    print("\n" + "=" * 65)
    print("  EGS -- FEATURE-ONLY DRIFT ANALYSIS")
    print("=" * 65)

    all_dfs = {}

    benign_path = os.path.join(EGS_DIR, "unified_egs_benign.csv")
    if os.path.exists(benign_path):
        all_dfs["benign"] = pd.read_csv(benign_path)
        print(f"  Loaded benign baseline: {benign_path}")
    else:
        print(f"  Warning: benign baseline not found at {benign_path}")
        print("  Run compute_unified_egs.py first for comparison.")

    for intensity in INTENSITIES:
        mode    = f"feature_drift_{intensity}"
        records = compute_egs_for_feature_drift(intensity)
        df      = add_unified_egs(pd.DataFrame(records))
        out     = os.path.join(EGS_DIR, f"egs_timeseries_{mode}.csv")
        df.to_csv(out, index=False)
        print(f"  Saved -> {out}")
        all_dfs[mode] = df

    cv_df = build_cv_table(all_dfs)
    cv_df.to_csv(os.path.join(EGS_DIR, "egs_feature_drift_cv_comparison.csv"),
                 index=False)

    print("\n" + "=" * 65)
    print("  CV COMPARISON -- Benign vs Feature Drift")
    print(f"\n{cv_df.to_string(index=False)}")
    print("=" * 65)

    plot_feature_drift_overlay(all_dfs)

    print("\n" + "=" * 65)
    print("  DONE -- Next: run compute_sci_feature_drift.py")
    print(f"  Outputs -> {EGS_DIR}")
    print("=" * 65)
    return all_dfs


if __name__ == "__main__":
    run_egs_feature_drift()