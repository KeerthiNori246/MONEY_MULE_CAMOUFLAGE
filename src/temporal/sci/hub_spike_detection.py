"""
hub_spike_detection.py
-----------------------
Applies spike detection to HR_t time-series from hub_redistribution.py.

    Spike at t if HR_t > μ_HR + k * σ_HR

Runs on benign and adversarial CSVs and compares:
    - Spike frequency
    - Spike magnitude
    - CV (coefficient of variation)

Also produces per-component plots (ΔD_t, ΔC_t, HR_t) mirroring
the structure of spike_detection.py for motifs.

Design mirrors spike_detection.py exactly so run_motif_pipeline.py
can call both the same way.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "graphs", "sci_results")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core spike detection (same formula as spike_detection.py)
# ---------------------------------------------------------------------------

def detect_spikes(series: pd.Series, k: float = 2.0):
    """
    Flag spikes in a time-series using μ + k*σ threshold.
    Identical to the same function in spike_detection.py.
    """
    mu        = series.mean()
    sigma     = series.std()
    threshold = mu + k * sigma
    spike_mask = series > threshold
    return spike_mask, threshold


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_hub_spikes(df: pd.DataFrame, mode: str, k: float = 2.0):
    """
    Run spike detection on delta_entropy, delta_centrality, and HR_t.

    Parameters
    ----------
    df   : pd.DataFrame  output from hub_redistribution.py
    mode : str           "benign" or "adv"
    k    : float         spike threshold multiplier

    Returns
    -------
    spike_summary : dict  per-signal spike stats
    df            : pd.DataFrame  with spike_* columns appended
    """
    print(f"\n{'='*50}")
    print(f"HUB SPIKE ANALYSIS — {mode.upper()}  (k={k})")
    print(f"{'='*50}")

    signal_cols  = ["delta_entropy", "delta_centrality", "HR_t"]
    spike_summary = {}

    for col in signal_cols:
        series = df[col]
        spike_mask, threshold = detect_spikes(series, k=k)

        n_spikes   = spike_mask.sum()
        spike_rate = n_spikes / len(series)
        mean_spike = series[spike_mask].mean() if n_spikes > 0 else 0
        max_spike  = series[spike_mask].max()  if n_spikes > 0 else 0
        cv         = (series.std() / series.mean()) if series.mean() != 0 else 0

        spike_summary[col] = {
            "threshold":             round(threshold, 4),
            "n_spikes":              int(n_spikes),
            "spike_rate":            round(spike_rate, 4),
            "mean_spike_magnitude":  round(mean_spike, 4),
            "max_spike":             round(max_spike,  4),
            "cv":                    round(cv, 4),
            "spike_dates":           df.loc[spike_mask, "date"].tolist()
        }

        print(f"\n  {col}:")
        print(f"    Threshold (μ + {k}σ) : {threshold:.4f}")
        print(f"    Spikes detected     : {n_spikes} / {len(series)} days")
        print(f"    Spike rate          : {spike_rate:.1%}")
        print(f"    Mean spike size     : {mean_spike:.4f}")
        print(f"    Max spike           : {max_spike:.4f}")
        print(f"    CV (σ/μ)            : {cv:.4f}")

        df[f"spike_{col}"] = spike_mask

    return spike_summary, df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_hub_comparison(df_benign, df_adv, signal="HR_t", k=2.0):
    """
    Plot a single hub signal (HR_t, delta_entropy, or delta_centrality)
    for benign vs adversarial on the same figure.
    Highlights spike points.
    Mirrors plot_motif_comparison() in spike_detection.py.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    label_map = {
        "HR_t":             "HR_t (Unified Hub Signal)",
        "delta_entropy":    "ΔD_t (Degree Distribution Entropy)",
        "delta_centrality": "ΔC_t (Top-k Centrality Change)",
    }
    fig.suptitle(f"Hub Redistribution: {label_map.get(signal, signal)}  (k={k})",
                 fontsize=14, fontweight="bold")

    for ax, df, mode, color in zip(
        axes,
        [df_benign, df_adv],
        ["Benign", "Adversarial"],
        ["steelblue", "crimson"]
    ):
        x      = range(len(df))
        values = df[signal].values

        _, threshold = detect_spikes(df[signal], k=k)
        spike_mask   = df[signal] > threshold

        ax.plot(x, values, color=color, linewidth=1.0, alpha=0.8,
                label=signal)
        ax.axhline(threshold, color="orange", linestyle="--",
                   linewidth=1.2, label=f"Threshold (μ+{k}σ)={threshold:.3f}")

        spike_x = [i for i, s in enumerate(spike_mask) if s]
        spike_y = [values[i] for i in spike_x]
        ax.scatter(spike_x, spike_y, color="black", zorder=5,
                   s=30, label=f"Spikes ({len(spike_x)})")

        ax.set_title(f"{mode} Baseline", fontsize=11)
        ax.set_ylabel(signal)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"hub_{signal}_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


def plot_hr_components(df, mode="benign", k=2.0):
    """
    Three-panel plot showing ΔD_t, ΔC_t, and HR_t stacked vertically
    for a single baseline. Useful for seeing how the components
    combine into the unified signal.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    color_map = {"benign": "steelblue", "adv": "crimson"}
    color     = color_map.get(mode, "steelblue")

    signals = ["delta_entropy", "delta_centrality", "HR_t"]
    labels  = ["ΔD_t (Entropy)", "ΔC_t (Centrality)", "HR_t (Unified)"]

    fig.suptitle(f"Hub Redistribution Components — {mode.upper()}",
                 fontsize=14, fontweight="bold")

    for ax, sig, lbl in zip(axes, signals, labels):
        x      = range(len(df))
        values = df[sig].values

        _, threshold = detect_spikes(df[sig], k=k)
        spike_mask   = df[sig] > threshold

        ax.plot(x, values, color=color, linewidth=1.0, alpha=0.85, label=lbl)
        ax.axhline(threshold, color="orange", linestyle="--",
                   linewidth=1.0, label=f"Threshold={threshold:.3f}")

        spike_x = [i for i, s in enumerate(spike_mask) if s]
        spike_y = [values[i] for i in spike_x]
        ax.scatter(spike_x, spike_y, color="black", zorder=5, s=25,
                   label=f"Spikes ({len(spike_x)})")

        ax.set_ylabel(lbl, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"hub_components_{mode}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_hub_spike_analysis(k=2.0):
    """
    Load hub CSVs and run full spike analysis + plots.
    Called from run_motif_pipeline.py as Phase 3.

    Returns
    -------
    summary_benign, summary_adv : dicts of per-signal spike stats
    """
    benign_csv = os.path.join(OUTPUT_DIR, "hub_timeseries_benign.csv")
    adv_csv    = os.path.join(OUTPUT_DIR, "hub_timeseries_adv.csv")

    if not os.path.exists(benign_csv) or not os.path.exists(adv_csv):
        raise FileNotFoundError(
            "Run hub_redistribution.py first to generate hub CSV files.\n"
            f"Expected:\n  {benign_csv}\n  {adv_csv}"
        )

    df_benign = pd.read_csv(benign_csv)
    df_adv    = pd.read_csv(adv_csv)

    # Spike analysis
    summary_benign, df_benign = analyze_hub_spikes(df_benign, "benign", k=k)
    summary_adv,    df_adv    = analyze_hub_spikes(df_adv,    "adv",    k=k)

    # Separability comparison table
    print(f"\n{'='*50}")
    print("HUB SEPARABILITY COMPARISON")
    print(f"{'='*50}")
    for col in ["delta_entropy", "delta_centrality", "HR_t"]:
        b = summary_benign[col]
        a = summary_adv[col]
        print(f"\n  {col}:")
        print(f"    Benign  — spike rate: {b['spike_rate']:.1%} | "
              f"mean magnitude: {b['mean_spike_magnitude']} | CV: {b['cv']:.4f}")
        print(f"    Advers  — spike rate: {a['spike_rate']:.1%} | "
              f"mean magnitude: {a['mean_spike_magnitude']} | CV: {a['cv']:.4f}")

    # Save spike-annotated CSVs
    df_benign.to_csv(os.path.join(OUTPUT_DIR, "hub_spikes_benign.csv"), index=False)
    df_adv.to_csv(   os.path.join(OUTPUT_DIR, "hub_spikes_adv.csv"),    index=False)
    print(f"\nSpike-annotated CSVs saved → {OUTPUT_DIR}")

    # Plots — comparison plots for each signal
    for signal in ["delta_entropy", "delta_centrality", "HR_t"]:
        plot_hub_comparison(df_benign, df_adv, signal=signal, k=k)

    # Component breakdown plots per baseline
    plot_hr_components(df_benign, mode="benign", k=k)
    plot_hr_components(df_adv,    mode="adv",    k=k)

    return summary_benign, summary_adv


if __name__ == "__main__":
    run_hub_spike_analysis(k=2.0)