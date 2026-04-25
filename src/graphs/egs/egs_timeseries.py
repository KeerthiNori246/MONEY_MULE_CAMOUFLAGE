"""
egs_timeseries.py
-----------------
Runs compute_egs_signals for both baselines and saves CSVs.

Output files:
    graphs/egs_results/egs_timeseries_benign.csv
    graphs/egs_results/egs_timeseries_adv.csv

CSV columns:
    date, n_tx, dispersion, delta_P, delta_D

Run this first, then egs_spike_detection.py.
"""

import os
import pandas as pd
from compute_egs_signals import compute_egs_signals, EGS_DIR


def save_timeseries(records: list, mode: str) -> pd.DataFrame:
    df       = pd.DataFrame(records)
    out_path = os.path.join(EGS_DIR, f"egs_timeseries_{mode}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    df_valid = df.dropna(subset=["delta_P"])

    print(f"  ── {mode.upper()} summary ──")
    for col in ["delta_P", "delta_D"]:
        mu  = df_valid[col].mean()
        std = df_valid[col].std()
        mx  = df_valid[col].max()
        cv  = std / mu if mu > 0 else float("nan")
        print(f"    {col}: mean={mu:.6f}  std={std:.6f}  max={mx:.6f}  CV={cv:.3f}")

    return df


def run_egs_timeseries():
    print("\n" + "=" * 60)
    print("  EGS TIMESERIES — Prototype Movement (ΔP) + Dispersion (ΔD)")
    print("=" * 60)

    results = {}
    for mode in ["benign", "adv"]:
        records       = compute_egs_signals(mode=mode)
        df            = save_timeseries(records, mode)
        results[mode] = df

    # CV comparison for both signals
    print("\n" + "=" * 60)
    print("  CV COMPARISON (Higher CV = more volatile)")
    print(f"  {'Signal':<10} {'Benign CV':>12} {'Adv CV':>12}")
    print("  " + "-" * 36)
    for col in ["delta_P", "delta_D"]:
        vals = {}
        for mode, df in results.items():
            s    = df[col].dropna()
            vals[mode] = s.std() / s.mean() if s.mean() > 0 else float("nan")
        print(f"  {col:<10} {vals['benign']:>12.3f} {vals['adv']:>12.3f}")

    print("\n  Next → run egs_spike_detection.py")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_egs_timeseries()