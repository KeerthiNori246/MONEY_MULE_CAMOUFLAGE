"""
egs_timeseries.py
-----------------
Runs compute_egs_signals for both baselines and saves CSVs.

Output files:
    graphs/egs_results/egs_timeseries_benign.csv
    graphs/egs_results/egs_timeseries_adv.csv

CSV columns:
    date, n_tx, delta_P

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

    # Summary stats (exclude day 0 NaN)
    df_valid = df.dropna(subset=["delta_P"])
    mu  = df_valid["delta_P"].mean()
    std = df_valid["delta_P"].std()
    mx  = df_valid["delta_P"].max()
    cv  = std / mu if mu > 0 else float("nan")

    print(f"  ── {mode.upper()} delta_P summary ──")
    print(f"    mean={mu:.6f}  std={std:.6f}  max={mx:.6f}  CV={cv:.3f}")

    return df


def run_egs_timeseries():
    print("\n" + "=" * 55)
    print("  EGS TIMESERIES — Prototype Center Movement (ΔP)")
    print("=" * 55)

    results = {}
    for mode in ["benign", "adv"]:
        records      = compute_egs_signals(mode=mode)
        df           = save_timeseries(records, mode)
        results[mode] = df

    # CV comparison
    print("\n" + "=" * 55)
    print("  CV COMPARISON — delta_P")
    print("  Higher CV = more volatile prototype movement")
    print("=" * 55)
    for mode, df in results.items():
        s  = df["delta_P"].dropna()
        cv = s.std() / s.mean() if s.mean() > 0 else float("nan")
        print(f"  {mode:8s} CV = {cv:.3f}")

    print("\n  Next → run egs_spike_detection.py")
    print("=" * 55)

    return results


if __name__ == "__main__":
    run_egs_timeseries()