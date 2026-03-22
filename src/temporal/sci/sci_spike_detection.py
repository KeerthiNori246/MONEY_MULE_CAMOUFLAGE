"""
sci_spike_detection.py
-----------------------
Applies spike detection to SCI_t and its component signals (ΔM_t, HR_t).

    Spike at t if SCI_t > μ_SCI + k · σ_SCI

Also produces the final separability comparison table and plots:
    - SCI_t benign vs adversarial (main result plot)
    - ΔM_t benign vs adversarial
    - HR_t benign vs adversarial (already produced by hub_spike_detection,
      reproduced here in the SCI context for completeness)
    - 3-panel component breakdown per baseline (ΔM_t / HR_t / SCI_t)

Design mirrors hub_spike_detection.py and spike_detection.py exactly.
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
# Core spike detection (same formula throughout the whole pipeline)
# ---------------------------------------------------------------------------

def detect_spikes(series: pd.Series, k: float = 2.0):
    """
    Flag spikes using μ + k·σ threshold.
    Identical to detect_spikes() in spike_detection.py and hub_spike_detection.py.
    """
    mu         = series.mean()
    sigma      = series.std()
    threshold  = mu + k * sigma
    spike_mask = series > threshold
    return spike_mask, threshold


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_sci_spikes(df: pd.DataFrame, mode: str, k: float = 2.0):
    """
    Run spike detection on delta_M_t, HR_t, and SCI_t.

    Parameters
    ----------
    df   : pd.DataFrame  output from compute_sci.py
    mode : str           "benign" or "adv"
    k    : float         spike threshold multiplier

    Returns
    -------
    spike_summary : dict
    df            : pd.DataFrame with spike_* columns appended
    """
    print(f"\n{'='*50}")
    print(f"SCI SPIKE ANALYSIS — {mode.upper()}  (k={k})")
    print(f"{'='*50}")

    signal_cols   = ["delta_M_t", "HR_t", "SCI_t"]
    spike_summary = {}

    for col in signal_cols:
        series = df[col]
        spike_mask, threshold = detect_spikes(series, k=k)

        n_spikes   = spike_mask.sum()
        spike_rate = n_spikes / len(series)
        mean_spike = series[spike_mask].mean() if n_spikes > 0 else 0
        max_spike  = series[spike_mask].max()  if n_spikes > 0 else 0
        # CV only meaningful when mean != 0
        mean_val   = series.mean()
        cv = (series.std() / mean_val) if mean_val != 0 else float("nan")

        spike_summary[col] = {
            "threshold":            round(threshold, 4),
            "n_spikes":             int(n_spikes),
            "spike_rate":           round(spike_rate, 4),
            "mean_spike_magnitude": round(mean_spike, 4),
            "max_spike":            round(max_spike,  4),
            "cv":                   round(cv, 4) if not np.isnan(cv) else "N/A",
            "spike_dates":          df.loc[spike_mask, "date"].tolist()
        }

        print(f"\n  {col}:")
        print(f"    Threshold (μ + {k}σ) : {threshold:.4f}")
        print(f"    Spikes detected     : {n_spikes} / {len(series)} days")
        print(f"    Spike rate          : {spike_rate:.1%}")
        print(f"    Mean spike size     : {mean_spike:.4f}")
        print(f"    Max spike           : {max_spike:.4f}")
        print(f"    CV (σ/μ)            : {cv:.4f}" if not np.isnan(cv) else "    CV (σ/μ)            : N/A (mean≈0)")

        df[f"spike_{col}"] = spike_mask

    return spike_summary, df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_sci_comparison(df_benign, df_adv, signal="SCI_t", k=2.0):
    """
    Benign vs adversarial comparison plot for a single signal.
    Mirrors plot_hub_comparison() in hub_spike_detection.py.
    """
    label_map = {
        "SCI_t":      "SCI_t (Structural Coordination Intensity)",
        "delta_M_t":  "ΔM_t (Unified Motif Signal)",
        "HR_t":       "HR_t (Unified Hub Signal)",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle(
        f"SCI — {label_map.get(signal, signal)}  (k={k})",
        fontsize=14, fontweight="bold"
    )

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

        ax.plot(x, values, color=color, linewidth=1.0, alpha=0.85, label=signal)
        ax.axhline(threshold, color="orange", linestyle="--",
                   linewidth=1.2, label=f"Threshold (μ+{k}σ)={threshold:.3f}")

        spike_x = [i for i, s in enumerate(spike_mask) if s]
        spike_y = [values[i] for i in spike_x]
        ax.scatter(spike_x, spike_y, color="black", zorder=5,
                   s=40, label=f"Spikes ({len(spike_x)})")

        ax.set_title(f"{mode} Baseline", fontsize=11)
        ax.set_ylabel(signal)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"sci_{signal}_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


def plot_sci_components(df, mode="benign", k=2.0):
    """
    3-panel plot: ΔM_t / HR_t / SCI_t stacked vertically for one baseline.
    Shows how the two components combine into the unified SCI signal.
    Mirrors plot_hr_components() in hub_spike_detection.py.
    """
    color_map = {"benign": "steelblue", "adv": "crimson"}
    color     = color_map.get(mode, "steelblue")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"SCI Components — {mode.upper()}",
        fontsize=14, fontweight="bold"
    )

    signals = ["delta_M_t", "HR_t",       "SCI_t"]
    labels  = ["ΔM_t (Motif)", "HR_t (Hub)", "SCI_t (Unified)"]

    for ax, sig, lbl in zip(axes, signals, labels):
        values = df[sig].values
        x      = range(len(values))

        _, threshold = detect_spikes(df[sig], k=k)
        spike_mask   = df[sig] > threshold

        ax.plot(x, values, color=color, linewidth=1.0, alpha=0.85, label=lbl)
        ax.axhline(threshold, color="orange", linestyle="--",
                   linewidth=1.0, label=f"Threshold={threshold:.3f}")

        spike_x = [i for i, s in enumerate(spike_mask) if s]
        spike_y = [values[i] for i in spike_x]
        ax.scatter(spike_x, spike_y, color="black", zorder=5,
                   s=25, label=f"Spikes ({len(spike_x)})")

        ax.set_ylabel(lbl, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"sci_components_{mode}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


def plot_sci_overlay(df_benign, df_adv, k=2.0):
    """
    Single-panel overlay plot of SCI_t for both baselines on the same axes.
    This is the main result plot — shows structural separation at a glance.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("SCI_t — Benign vs Adversarial Overlay", fontsize=14, fontweight="bold")

    for df, mode, color in [
        (df_benign, "Benign",       "steelblue"),
        (df_adv,    "Adversarial",  "crimson"),
    ]:
        x      = range(len(df))
        values = df["SCI_t"].values
        _, thr = detect_spikes(df["SCI_t"], k=k)
        spike_mask = df["SCI_t"] > thr

        ax.plot(x, values, color=color, linewidth=1.2, alpha=0.85, label=f"{mode} SCI_t")
        ax.axhline(thr, color=color, linestyle=":", linewidth=1.0, alpha=0.5,
                   label=f"{mode} threshold={thr:.2f}")

        spike_x = [i for i, s in enumerate(spike_mask) if s]
        spike_y = [values[i] for i in spike_x]
        ax.scatter(spike_x, spike_y, color=color, marker="*", zorder=5,
                   s=120, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SCI_t")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "sci_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sci_spike_analysis(k=2.0):
    """
    Load SCI CSVs and run full spike analysis + plots.
    Called from run_motif_pipeline.py as Phase 6.

    Returns
    -------
    summary_benign, summary_adv : dicts of per-signal spike stats
    """
    benign_csv = os.path.join(OUTPUT_DIR, "sci_timeseries_benign.csv")
    adv_csv    = os.path.join(OUTPUT_DIR, "sci_timeseries_adv.csv")

    if not os.path.exists(benign_csv) or not os.path.exists(adv_csv):
        raise FileNotFoundError(
            "Run compute_sci.py first to generate SCI CSV files.\n"
            f"Expected:\n  {benign_csv}\n  {adv_csv}"
        )

    df_benign = pd.read_csv(benign_csv)
    df_adv    = pd.read_csv(adv_csv)

    # Spike analysis per baseline
    summary_benign, df_benign = analyze_sci_spikes(df_benign, "benign", k=k)
    summary_adv,    df_adv    = analyze_sci_spikes(df_adv,    "adv",    k=k)

    # Final separability comparison table
    print(f"\n{'='*50}")
    print("FINAL SCI SEPARABILITY COMPARISON")
    print(f"{'='*50}")
    for col in ["delta_M_t", "HR_t", "SCI_t"]:
        b = summary_benign[col]
        a = summary_adv[col]
        print(f"\n  {col}:")
        print(f"    Benign  — spike rate: {b['spike_rate']:.1%} | "
              f"mean magnitude: {b['mean_spike_magnitude']} | CV: {b['cv']}")
        print(f"    Advers  — spike rate: {a['spike_rate']:.1%} | "
              f"mean magnitude: {a['mean_spike_magnitude']} | CV: {a['cv']}")

    # Save spike-annotated CSVs
    df_benign.to_csv(os.path.join(OUTPUT_DIR, "sci_spikes_benign.csv"), index=False)
    df_adv.to_csv(   os.path.join(OUTPUT_DIR, "sci_spikes_adv.csv"),    index=False)
    print(f"\nSpike-annotated CSVs saved → {OUTPUT_DIR}")

    # Plots
    for signal in ["delta_M_t", "HR_t", "SCI_t"]:
        plot_sci_comparison(df_benign, df_adv, signal=signal, k=k)

    plot_sci_components(df_benign, mode="benign", k=k)
    plot_sci_components(df_adv,    mode="adv",    k=k)
    plot_sci_overlay(df_benign, df_adv, k=k)   # ← main result plot

    return summary_benign, summary_adv


if __name__ == "__main__":
    run_sci_spike_analysis(k=2.0)