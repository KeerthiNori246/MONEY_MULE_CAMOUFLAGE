"""
compute_unified_egs.py
----------------------
Combines ΔP, ΔD, ΔA into a single Unified Embedding Geometry Signal.

    EGS^(t) = α · ΔP̂^(t) + β · ΔD̂^(t) + γ · ΔÂ^(t)

Where:
    ΔP̂, ΔD̂, ΔÂ = Z-score standardized versions of each signal
    α = β = γ = 1/3  (equal weights)

Z-scoring is computed independently per baseline — benign uses benign
statistics, adversarial uses adversarial statistics.

Reads:
    graphs/egs_results/egs_timeseries_benign.csv
    graphs/egs_results/egs_timeseries_adv.csv

Outputs:
    graphs/egs_results/unified_egs_benign.csv
    graphs/egs_results/unified_egs_adv.csv
    graphs/egs_results/unified_egs_spike_benign.png   ← per-baseline stacked
    graphs/egs_results/unified_egs_spike_adv.png
    graphs/egs_results/unified_egs_overlay.png        ← benign vs adv overlay
    graphs/egs_results/unified_egs_components.png     ← components + EGS_t
    graphs/egs_results/unified_egs_summary.csv        ← spike stats table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ───────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
EGS_DIR       = os.path.join(_PROJECT_ROOT, "graphs", "egs_results")

COLORS = {"benign": "#5B9BD5", "adv": "#E06C75"}
ALPHA  = 1 / 3   # equal weights


# ── helpers ──────────────────────────────────────────────────────────────

def zscore(series: pd.Series) -> pd.Series:
    """Z-score standardize a series. Returns zero series if std=0."""
    mu  = series.mean()
    std = series.std()
    if std < 1e-12:
        return series * 0.0
    return (series - mu) / std


def compute_unified_egs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a timeseries DataFrame (from egs_timeseries_*.csv),
    Z-score each delta signal independently and combine with equal weights.

    Returns the same DataFrame with two new columns:
        delta_P_z, delta_D_z, delta_A_z  — standardized components
        EGS_t                             — unified signal
    """
    # Work only on valid rows (drop day 0 NaN)
    valid_mask = df["delta_P"].notna()
    df = df.copy()

    for col in ["delta_P", "delta_D", "delta_A"]:
        z_col = f"{col}_z"
        df[z_col] = float("nan")
        df.loc[valid_mask, z_col] = zscore(df.loc[valid_mask, col]).values

    df["EGS_t"] = float("nan")
    df.loc[valid_mask, "EGS_t"] = (
        ALPHA * df.loc[valid_mask, "delta_P_z"] +
        ALPHA * df.loc[valid_mask, "delta_D_z"] +
        ALPHA * df.loc[valid_mask, "delta_A_z"]
    )

    return df


def detect_spikes(series: pd.Series, k: float = 2.0):
    mu        = series.mean()
    sigma     = series.std()
    threshold = mu + k * sigma
    spikes    = series > threshold
    return threshold, spikes


def spike_stats(series: pd.Series, k: float = 2.0) -> dict:
    threshold, spikes = detect_spikes(series, k)
    spike_count       = int(spikes.sum())
    spike_rate        = spike_count / len(series)
    mean_spike_mag    = float(series[spikes].mean()) if spike_count > 0 else 0.0
    cv                = float(series.std() / series.mean()) if series.mean() > 0 else float("nan")
    return {
        "threshold":      round(float(threshold), 6),
        "spike_count":    spike_count,
        "spike_rate":     round(spike_rate, 4),
        "mean_spike_mag": round(mean_spike_mag, 6),
        "CV":             round(cv, 4),
    }


# ── plots ────────────────────────────────────────────────────────────────

def plot_components_and_egs(df: pd.DataFrame, mode: str):
    """
    4-row figure per baseline:
      Row 1 — ΔP_z (standardized)
      Row 2 — ΔD_z (standardized)
      Row 3 — ΔA_z (standardized)
      Row 4 — EGS_t (unified)
    """
    valid = df.dropna(subset=["EGS_t"])
    x     = np.arange(len(valid))
    color = COLORS[mode]

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    fig.suptitle(
        f"EGS Components + Unified Signal — {mode.upper()}",
        fontsize=13, fontweight="bold"
    )

    component_cols = [
        ("delta_P_z", "ΔP̂_t (Z-scored Prototype Movement)"),
        ("delta_D_z", "ΔD̂_t (Z-scored Dispersion Change)"),
        ("delta_A_z", "ΔÂ_t (Z-scored Anisotropy Change)"),
        ("EGS_t",     "EGS_t (Unified)"),
    ]

    for ax, (col, label) in zip(axes, component_cols):
        series            = valid[col]
        threshold, spikes = detect_spikes(series)
        stats             = spike_stats(series)

        lw    = 1.8 if col == "EGS_t" else 1.2
        alpha = 1.0 if col == "EGS_t" else 0.85

        ax.plot(x, series.values, color=color, linewidth=lw,
                alpha=alpha, label=col)
        ax.axhline(threshold, color="orange", linestyle="--",
                   linewidth=1.0,
                   label=f"Threshold={threshold:.3f}")
        ax.scatter(x[spikes.values], series.values[spikes.values],
                   color="black", zorder=5, s=40,
                   label=f"Spikes ({stats['spike_count']})")

        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(EGS_DIR, f"unified_egs_spike_{mode}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


def plot_overlay(df_benign: pd.DataFrame, df_adv: pd.DataFrame, k: float = 2.0):
    """
    Single panel: benign EGS_t vs adversarial EGS_t overlaid.
    The main separability figure for the paper.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        "EGS_t — Benign vs Adversarial Overlay",
        fontsize=13, fontweight="bold"
    )

    for mode, df in [("benign", df_benign), ("adv", df_adv)]:
        valid  = df.dropna(subset=["EGS_t"])
        series = valid["EGS_t"]
        x      = np.arange(len(series))
        stats  = spike_stats(series, k)
        color  = COLORS[mode]

        ax.plot(x, series.values, color=color, linewidth=1.4,
                label=f"{mode.capitalize()} (CV={stats['CV']:.3f})",
                alpha=0.9)
        ax.axhline(stats["threshold"], color=color, linestyle=":",
                   linewidth=0.8, alpha=0.6,
                   label=f"{mode} threshold={stats['threshold']:.3f}")

        # Mark spikes
        _, spikes = detect_spikes(series, k)
        ax.scatter(x[spikes.values], series.values[spikes.values],
                   color=color, marker="*", s=100, zorder=5)

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("EGS_t")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EGS_DIR, "unified_egs_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


def plot_combined_components(df_benign: pd.DataFrame, df_adv: pd.DataFrame):
    """
    4-row figure with both baselines overlaid per row.
    Rows: ΔP_z, ΔD_z, ΔA_z, EGS_t
    Paper-ready summary figure.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    fig.suptitle(
        "EGS — Standardized Components + Unified Signal\nBenign vs Adversarial",
        fontsize=13, fontweight="bold"
    )

    rows = [
        ("delta_P_z", "ΔP̂_t"),
        ("delta_D_z", "ΔD̂_t"),
        ("delta_A_z", "ΔÂ_t"),
        ("EGS_t",     "EGS_t (Unified)"),
    ]

    for ax, (col, label) in zip(axes, rows):
        for mode, df in [("benign", df_benign), ("adv", df_adv)]:
            valid  = df.dropna(subset=[col])
            series = valid[col]
            x      = np.arange(len(series))
            stats  = spike_stats(series)
            color  = COLORS[mode]

            lw = 1.6 if col == "EGS_t" else 1.1
            ax.plot(x, series.values, color=color, linewidth=lw,
                    label=f"{mode.capitalize()} CV={stats['CV']:.3f}",
                    alpha=0.85)

            _, spikes = detect_spikes(series)
            ax.scatter(x[spikes.values], series.values[spikes.values],
                       color=color, marker="*", s=60, zorder=5)

        ax.axhline(0, color="gray", linestyle="-",
                   linewidth=0.4, alpha=0.4)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(EGS_DIR, "unified_egs_components.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────

def run_unified_egs(k: float = 2.0):
    benign_path = os.path.join(EGS_DIR, "egs_timeseries_benign.csv")
    adv_path    = os.path.join(EGS_DIR, "egs_timeseries_adv.csv")

    if not os.path.exists(benign_path) or not os.path.exists(adv_path):
        raise FileNotFoundError(
            "Component CSVs not found. Run egs_timeseries.py first."
        )

    print("\n" + "=" * 65)
    print("  UNIFIED EGS SIGNAL")
    print(f"  EGS_t = (1/3)·ΔP̂ + (1/3)·ΔD̂ + (1/3)·ΔÂ  (k={k})")
    print("=" * 65)

    df_benign_raw = pd.read_csv(benign_path)
    df_adv_raw    = pd.read_csv(adv_path)

    df_benign = compute_unified_egs(df_benign_raw)
    df_adv    = compute_unified_egs(df_adv_raw)

    # Save unified CSVs
    for mode, df in [("benign", df_benign), ("adv", df_adv)]:
        out = os.path.join(EGS_DIR, f"unified_egs_{mode}.csv")
        df.to_csv(out, index=False)
        print(f"\n  Saved → {out}")

    # Spike stats for EGS_t
    print("\n" + "=" * 65)
    print("  EGS_t SPIKE DETECTION")
    print("=" * 65)

    summary_rows = []
    for mode, df in [("benign", df_benign), ("adv", df_adv)]:
        series = df["EGS_t"].dropna()
        stats  = spike_stats(series, k)
        print(
            f"  [{mode:8s}] threshold={stats['threshold']:.4f}  "
            f"spikes={stats['spike_count']}  "
            f"rate={stats['spike_rate']:.1%}  "
            f"mean_mag={stats['mean_spike_mag']:.4f}  "
            f"CV={stats['CV']:.3f}"
        )
        summary_rows.append({"mode": mode, **stats})

    # CV comparison across all signals
    print("\n" + "=" * 65)
    print("  FULL CV SUMMARY — All Signals")
    print(f"  {'Signal':<12} {'Benign CV':>12} {'Adv CV':>12}")
    print("  " + "-" * 38)
    all_signals = ["delta_P", "delta_D", "delta_A", "EGS_t"]
    for col in all_signals:
        cvs = {}
        for mode, df in [("benign", df_benign), ("adv", df_adv)]:
            s = df[col].dropna()
            cvs[mode] = s.std() / s.mean() if s.mean() > 0 else float("nan")
        print(f"  {col:<12} {cvs['benign']:>12.3f} {cvs['adv']:>12.3f}")

    # Plots
    print("\n  Generating plots...")
    plot_components_and_egs(df_benign, mode="benign")
    plot_components_and_egs(df_adv,    mode="adv")
    plot_overlay(df_benign, df_adv, k=k)
    plot_combined_components(df_benign, df_adv)

    # Save summary
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(EGS_DIR, "unified_egs_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary → {summary_path}")

    print("\n" + "=" * 65)
    print("  EGS PIPELINE COMPLETE")
    print("  ΔP + ΔD + ΔA → unified EGS_t")
    print(f"  All outputs → {EGS_DIR}")
    print("=" * 65)

    return df_benign, df_adv


if __name__ == "__main__":
    run_unified_egs()