"""
egs_spike_detection.py
----------------------
Spike analysis and visualization on ΔP, ΔD, ΔA signals.

Same spike logic as SCI pipeline:
    spike at t if value > μ + k·σ   (k=2.0 default)

Reads:
    graphs/egs_results/egs_timeseries_benign.csv
    graphs/egs_results/egs_timeseries_adv.csv

Outputs:
    graphs/egs_results/egs_spike_delta_P.png
    graphs/egs_results/egs_spike_delta_D.png
    graphs/egs_results/egs_spike_delta_A.png
    graphs/egs_results/egs_overlay_delta_P.png
    graphs/egs_results/egs_overlay_delta_D.png
    graphs/egs_results/egs_overlay_delta_A.png
    graphs/egs_results/egs_components_combined.png   ← 3-row paper figure
    graphs/egs_results/egs_spike_summary.csv
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

SIGNAL_LABELS = {
    "delta_P": "ΔP_t (Prototype Movement)",
    "delta_D": "ΔD_t (Dispersion Change)",
    "delta_A": "ΔA_t (Anisotropy Change)",
}


# ── spike helpers ────────────────────────────────────────────────────────

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
        "threshold":      round(threshold, 6),
        "spike_count":    spike_count,
        "spike_rate":     round(spike_rate, 4),
        "mean_spike_mag": round(mean_spike_mag, 6),
        "CV":             round(cv, 4),
    }


# ── plots ────────────────────────────────────────────────────────────────

def plot_stacked(df_benign, df_adv, col, k=2.0):
    """Two-panel plot: benign on top, adversarial below."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle(
        f"EGS — {SIGNAL_LABELS[col]}  (k={k})",
        fontsize=13, fontweight="bold"
    )

    for ax, (mode, df) in zip(axes, [("benign", df_benign), ("adv", df_adv)]):
        series            = df[col].dropna()
        x                 = np.arange(len(series))
        threshold, spikes = detect_spikes(series, k)
        stats             = spike_stats(series, k)
        color             = COLORS[mode]

        ax.plot(x, series.values, color=color, linewidth=1.2, label=col)
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.0,
                   label=f"Threshold (μ+{k}σ)={threshold:.6f}")
        ax.scatter(x[spikes.values], series.values[spikes.values],
                   color="black", zorder=5, s=40,
                   label=f"Spikes ({stats['spike_count']})")

        ax.set_title(
            f"{mode.capitalize()} Baseline   "
            f"[CV={stats['CV']:.3f}  spikes={stats['spike_count']}  "
            f"rate={stats['spike_rate']:.1%}]",
            fontsize=10
        )
        ax.set_ylabel(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(EGS_DIR, f"egs_spike_{col}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


def plot_overlay(df_benign, df_adv, col, k=2.0):
    """Single panel with both baselines overlaid."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        f"EGS — {SIGNAL_LABELS[col]} — Benign vs Adversarial Overlay",
        fontsize=12, fontweight="bold"
    )

    for mode, df in [("benign", df_benign), ("adv", df_adv)]:
        series = df[col].dropna()
        x      = np.arange(len(series))
        stats  = spike_stats(series, k)
        color  = COLORS[mode]

        ax.plot(x, series.values, color=color, linewidth=1.2,
                label=f"{mode.capitalize()} (CV={stats['CV']:.3f})",
                alpha=0.85)
        ax.axhline(stats["threshold"], color=color, linestyle=":",
                   linewidth=0.8, alpha=0.6,
                   label=f"{mode} threshold={stats['threshold']:.6f}")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(col)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EGS_DIR, f"egs_overlay_{col}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


def plot_combined(df_benign, df_adv, k=2.0):
    """
    3-row subplot — one row per signal (ΔP, ΔD, ΔA).
    Both baselines overlaid in each panel.
    Main EGS components figure for the paper.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(
        "EGS Components — Benign vs Adversarial",
        fontsize=13, fontweight="bold"
    )

    for ax, col in zip(axes, ["delta_P", "delta_D", "delta_A"]):
        for mode, df in [("benign", df_benign), ("adv", df_adv)]:
            series = df[col].dropna()
            x      = np.arange(len(series))
            stats  = spike_stats(series, k)
            color  = COLORS[mode]

            ax.plot(x, series.values, color=color, linewidth=1.2,
                    label=f"{mode.capitalize()} CV={stats['CV']:.3f}",
                    alpha=0.85)

            _, spikes = detect_spikes(series, k)
            ax.scatter(x[spikes.values], series.values[spikes.values],
                       color=color, marker="*", s=80, zorder=5)

        ax.set_ylabel(SIGNAL_LABELS[col], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()

    out = os.path.join(EGS_DIR, "egs_components_combined.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────

def run_egs_spike_detection(k: float = 2.0):
    benign_path = os.path.join(EGS_DIR, "egs_timeseries_benign.csv")
    adv_path    = os.path.join(EGS_DIR, "egs_timeseries_adv.csv")

    if not os.path.exists(benign_path) or not os.path.exists(adv_path):
        raise FileNotFoundError(
            "CSVs not found. Run egs_timeseries.py first."
        )

    df_benign = pd.read_csv(benign_path)
    df_adv    = pd.read_csv(adv_path)

    print("\n" + "=" * 65)
    print(f"  EGS SPIKE DETECTION  (k={k})")
    print("=" * 65)

    summary_rows = []

    for col in ["delta_P", "delta_D", "delta_A"]:
        print(f"\n  ── {col} ──")
        for mode, df in [("benign", df_benign), ("adv", df_adv)]:
            series = df[col].dropna()
            stats  = spike_stats(series, k)
            print(
                f"    [{mode:8s}] threshold={stats['threshold']:.6f}  "
                f"spikes={stats['spike_count']}  "
                f"rate={stats['spike_rate']:.1%}  "
                f"mean_mag={stats['mean_spike_mag']:.6f}  "
                f"CV={stats['CV']:.3f}"
            )
            summary_rows.append({"signal": col, "mode": mode, **stats})

        plot_stacked(df_benign, df_adv, col, k=k)
        plot_overlay(df_benign, df_adv, col, k=k)

    plot_combined(df_benign, df_adv, k=k)

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(EGS_DIR, "egs_spike_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary → {summary_path}")

    print("\n" + "=" * 65)
    print("  DONE")
    print(f"  Plots → {EGS_DIR}")
    print("=" * 65)

    return summary_df


if __name__ == "__main__":
    run_egs_spike_detection()