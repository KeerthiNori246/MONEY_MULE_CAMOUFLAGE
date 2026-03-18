"""
spike_detection.py
------------------
Applies the spike detection formula to ΔM_t time-series:

    Spike at t if ΔM_t > μ_ΔM + k * σ_ΔM

Runs on benign and adversarial CSVs and compares:
- Spike frequency (how often spikes occur)
- Spike magnitude (how large the spikes are)
- Spike concentration (are adversarial spikes clustered?)

This is the statistical separability test.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for all environments
import matplotlib.pyplot as plt

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "graphs", "sci_results")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def detect_spikes(series: pd.Series, k: float = 2.0):
    """
    Flag spikes in a ΔM time-series.

    Parameters
    ----------
    series : pd.Series
        The delta time-series (e.g., delta_triangles)
    k : float
        Sensitivity multiplier. k=2 means 2 std above mean.
        Lower k = more sensitive. Recommended: 2.0

    Returns
    -------
    spike_mask : pd.Series (bool)
        True at timesteps where a spike occurs
    threshold : float
        The computed threshold value
    """
    mu    = series.mean()
    sigma = series.std()
    threshold = mu + k * sigma
    spike_mask = series > threshold
    return spike_mask, threshold


def analyze_spikes(df: pd.DataFrame, mode: str, k: float = 2.0):
    """
    Run spike detection on all three delta columns and print summary.

    Parameters
    ----------
    df : pd.DataFrame
        Output from temporal_motif_evolution.py
    mode : str
        "benign" or "adv" (for labeling)
    k : float
        Spike threshold multiplier

    Returns
    -------
    spike_summary : dict
        Per-motif spike stats
    """
    print(f"\n{'='*50}")
    print(f"SPIKE ANALYSIS — {mode.upper()}  (k={k})")
    print(f"{'='*50}")

    delta_cols = ["delta_triangles", "delta_wedges"]
    spike_summary = {}

    for col in delta_cols:
        series = df[col]
        spike_mask, threshold = detect_spikes(series, k=k)

        n_spikes     = spike_mask.sum()
        spike_rate   = n_spikes / len(series)
        mean_spike   = series[spike_mask].mean() if n_spikes > 0 else 0
        max_spike    = series[spike_mask].max()  if n_spikes > 0 else 0
        # CV = σ/μ — measures relative volatility across all timesteps
        # High CV = signal is restless/bursty. Low CV = signal is stable.
        cv = (series.std() / series.mean()) if series.mean() > 0 else 0

        spike_summary[col] = {
            "threshold":   round(threshold, 4),
            "n_spikes":    int(n_spikes),
            "spike_rate":  round(spike_rate, 4),
            "mean_spike_magnitude": round(mean_spike, 4),
            "max_spike":   round(max_spike, 4),
            "cv":          round(cv, 4),
            "spike_dates": df.loc[spike_mask, "date"].tolist()
        }

        print(f"\n  {col}:")
        print(f"    Threshold (μ + {k}σ) : {threshold:.4f}")
        print(f"    Spikes detected     : {n_spikes} / {len(series)} days")
        print(f"    Spike rate          : {spike_rate:.1%}")
        print(f"    Mean spike size     : {mean_spike:.4f}")
        print(f"    Max spike           : {max_spike:.4f}")
        print(f"    CV (σ/μ)            : {cv:.4f}")

    # Add spike columns to df (for plotting)
    for col in delta_cols:
        mask, _ = detect_spikes(df[col], k=k)
        df[f"spike_{col}"] = mask

    return spike_summary, df


def plot_motif_comparison(df_benign, df_adv, motif="triangles", k=2.0):
    """
    Plot ΔM_t for benign vs adversarial on the same axes.
    Highlights spike points.

    Parameters
    ----------
    df_benign : pd.DataFrame
    df_adv    : pd.DataFrame
    motif     : str  — "triangles", "wedges", or "flf_chains"
    k         : float
    """
    col        = f"delta_{motif}"
    spike_col  = f"spike_{col}"

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle(f"Temporal Motif Evolution: Δ{motif.capitalize()}  (k={k})",
                 fontsize=14, fontweight="bold")

    for ax, df, mode, color in zip(
        axes,
        [df_benign, df_adv],
        ["Benign", "Adversarial"],
        ["steelblue", "crimson"]
    ):
        x     = range(len(df))
        delta = df[col].values

        # Compute threshold for this series
        _, threshold = detect_spikes(df[col], k=k)
        spike_mask   = df[col] > threshold

        ax.plot(x, delta, color=color, linewidth=1.0, alpha=0.8, label=f"Δ{motif}")
        ax.axhline(threshold, color="orange", linestyle="--",
                   linewidth=1.2, label=f"Threshold (μ+{k}σ)={threshold:.2f}")

        # Mark spike points
        spike_x = [i for i, s in enumerate(spike_mask) if s]
        spike_y = [delta[i] for i in spike_x]
        ax.scatter(spike_x, spike_y, color="black", zorder=5,
                   s=30, label=f"Spikes ({len(spike_x)})")

        ax.set_title(f"{mode} Baseline", fontsize=11)
        ax.set_ylabel(f"Δ{motif}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = f"{PLOT_DIR}/delta_{motif}_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


def run_spike_analysis(k=2.0):
    """
    Load CSVs from temporal_motif_evolution.py output and run full analysis.
    """
    benign_csv = f"{OUTPUT_DIR}/motif_timeseries_benign.csv"
    adv_csv    = f"{OUTPUT_DIR}/motif_timeseries_adv.csv"

    if not os.path.exists(benign_csv) or not os.path.exists(adv_csv):
        raise FileNotFoundError(
            "Run temporal_motif_evolution.py first to generate the CSV files."
        )

    df_benign = pd.read_csv(benign_csv)
    df_adv    = pd.read_csv(adv_csv)

    # Spike analysis
    summary_benign, df_benign = analyze_spikes(df_benign, "benign", k=k)
    summary_adv,    df_adv    = analyze_spikes(df_adv,    "adv",    k=k)

    # Side-by-side comparison
    print(f"\n{'='*50}")
    print("SEPARABILITY COMPARISON")
    print(f"{'='*50}")
    for col in ["delta_triangles", "delta_wedges"]:
        b = summary_benign[col]
        a = summary_adv[col]
        print(f"\n  {col}:")
        print(f"    Benign  spike rate: {b['spike_rate']:.1%}  | mean magnitude: {b['mean_spike_magnitude']}  | CV: {b['cv']:.4f}")
        print(f"    Advers  spike rate: {a['spike_rate']:.1%}  | mean magnitude: {a['mean_spike_magnitude']}  | CV: {a['cv']:.4f}")

    # Save spike-annotated CSVs
    df_benign.to_csv(f"{OUTPUT_DIR}/spikes_benign.csv", index=False)
    df_adv.to_csv(   f"{OUTPUT_DIR}/spikes_adv.csv",    index=False)

    # Plots
    for motif in ["triangles", "wedges"]:
        plot_motif_comparison(df_benign, df_adv, motif=motif, k=k)

    return summary_benign, summary_adv


if __name__ == "__main__":
    run_spike_analysis(k=2.0)