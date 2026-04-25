"""
compute_egs_signals.py
----------------------
Computes per-day EGS sub-signals for a single baseline (benign or adv).

    ΔP_t — Prototype Center Movement
            C_t  = (1 / n_t) * sum of all embedding vectors on day t
            ΔP_t = || C_t - C_{t-1} ||_2

    ΔD_t — Cluster Dispersion Change
            D_t  = (1 / n_t) * sum of || x_i - C_t || for all i
            ΔD_t = | D_t - D_{t-1} |

Day 0 has no previous day so delta_P and delta_D are NaN.

Usage
-----
    from compute_egs_signals import compute_egs_signals
    records = compute_egs_signals(mode="benign")
"""

import os
import glob
import pickle
import numpy as np

# ── paths ───────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

EMBED_DIR = os.path.join(_PROJECT_ROOT, "graphs", "embeddings")
EGS_DIR   = os.path.join(_PROJECT_ROOT, "graphs", "egs_results")
os.makedirs(EGS_DIR, exist_ok=True)


def compute_prototype(emb: np.ndarray) -> np.ndarray:
    """
    C_t = mean of all embedding rows.
    emb shape : (n_tx, 64)
    returns   : (64,) prototype vector
    """
    return emb.mean(axis=0)


def compute_dispersion(emb: np.ndarray, prototype: np.ndarray) -> float:
    """
    D_t = mean L2 distance of each node from its prototype.

        D_t = (1 / n_t) * sum_i || x_i - C_t ||

    emb shape       : (n_tx, 64)
    prototype shape : (64,)
    returns         : scalar dispersion value
    """
    diffs = emb - prototype                     # (n_tx, 64)
    dists = np.linalg.norm(diffs, axis=1)       # (n_tx,)
    return float(dists.mean())


def compute_egs_signals(mode: str = "benign") -> list:
    """
    Iterate all daily .npy files for a baseline in date order.
    For each consecutive pair compute ΔP_t and ΔD_t.

    Parameters
    ----------
    mode : "benign" or "adv"

    Returns
    -------
    List of dicts with keys: date, n_tx, dispersion, delta_P, delta_D
    """
    embed_dir = os.path.join(EMBED_DIR, mode)
    npy_files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in {embed_dir}\n"
            f"Run embed_snapshots.py first."
        )

    print(f"\n{'='*60}")
    print(f"  EGS — ΔP + ΔD — {mode.upper()}")
    print(f"  Found {len(npy_files)} daily embedding files")
    print(f"{'='*60}")

    records   = []
    prev_proto = None
    prev_disp  = None

    for npy_path in npy_files:
        date_str  = os.path.basename(npy_path).replace(".npy", "")
        meta_path = os.path.join(embed_dir, f"meta_{date_str}.pkl")

        emb = np.load(npy_path)           # (n_tx, 64)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        n_tx  = emb.shape[0]
        proto = compute_prototype(emb)    # (64,)
        disp  = compute_dispersion(emb, proto)  # scalar

        if prev_proto is None:
            delta_P = float("nan")
            delta_D = float("nan")
        else:
            delta_P = float(np.linalg.norm(proto - prev_proto))
            delta_D = float(abs(disp - prev_disp))

        records.append({
            "date":       date_str,
            "n_tx":       n_tx,
            "dispersion": disp,
            "delta_P":    delta_P,
            "delta_D":    delta_D,
        })

        print(
            f"  {date_str} | n={n_tx:5d} | "
            f"D={disp:.6f} | "
            f"ΔP={delta_P:.6f} | "
            f"ΔD={delta_D:.6f}"
        )

        prev_proto = proto
        prev_disp  = disp

    print(f"\n  Done. {len(records)} records computed.")
    return records