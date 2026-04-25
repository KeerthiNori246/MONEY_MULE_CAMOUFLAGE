"""
compute_egs_signals.py
----------------------
Computes per-day EGS sub-signals for a single baseline (benign or adv).

    ΔP_t — Prototype Center Movement
            C_t  = (1 / n_t) * sum of all embedding vectors on day t
            ΔP_t = || C_t - C_{t-1} ||_2

    ΔD_t — Cluster Dispersion Change
            D_t  = (1 / n_t) * sum_i || x_i - C_t ||
            ΔD_t = | D_t - D_{t-1} |

    ΔA_t — Anisotropy Change
            Σ_t  = (1 / n_t) * (X - C_t)^T @ (X - C_t)   [64x64 covariance]
            λ_1 ≥ λ_2 ≥ ... ≥ λ_d  (eigenvalues of Σ_t)
            A_t  = λ_1 / sum(λ_j)   [fraction of variance in top direction]
            ΔA_t = | A_t - A_{t-1} |

Day 0 has no previous day so delta_P, delta_D, delta_A are all NaN.

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


# ── per-day geometry ─────────────────────────────────────────────────────

def compute_prototype(emb: np.ndarray) -> np.ndarray:
    """
    C_t = mean of all embedding rows.
    emb shape : (n_tx, 64)
    returns   : (64,) prototype vector
    """
    return emb.mean(axis=0)


def compute_dispersion(emb: np.ndarray, prototype: np.ndarray) -> float:
    """
    D_t = (1 / n_t) * sum_i || x_i - C_t ||
    emb shape       : (n_tx, 64)
    prototype shape : (64,)
    returns         : scalar
    """
    diffs = emb - prototype                  # (n_tx, 64)
    dists = np.linalg.norm(diffs, axis=1)    # (n_tx,)
    return float(dists.mean())


def compute_anisotropy(emb: np.ndarray, prototype: np.ndarray) -> float:
    """
    A_t = λ_1 / sum(λ_j)

    Steps:
      1. Center the embeddings: X_c = emb - C_t
      2. Build covariance matrix: Σ = (1/n) * X_c^T @ X_c  → shape (64, 64)
      3. Compute eigenvalues of Σ
      4. A_t = largest eigenvalue / sum of all eigenvalues

    Edge cases:
      - n < 2  : return uniform baseline 1/d  (degenerate, e.g. day 31 benign n=1)
      - all λ≈0: return uniform baseline 1/d  (all embeddings identical)

    emb shape       : (n_tx, 64)
    prototype shape : (64,)
    returns         : scalar in (0, 1]
    """
    n, d = emb.shape

    # degenerate: single node or all identical
    if n < 2:
        return 1.0 / d

    centered = emb - prototype               # (n_tx, 64)

    # covariance matrix (64 x 64)
    # np.cov expects (d, n) — transpose centered
    if n >= d:
        # standard path: full covariance matrix
        cov = np.cov(centered.T)             # (64, 64)
        eigenvalues = np.linalg.eigvalsh(cov)   # sorted ascending, real
    else:
        # fewer samples than dimensions — use SVD for numerical stability
        # singular values s relate to eigenvalues by λ = s^2 / (n-1)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s ** 2) / (n - 1)

    eigenvalues = np.abs(eigenvalues)        # numerical safety
    total = eigenvalues.sum()

    if total < 1e-12:
        return 1.0 / d   # all embeddings identical

    lambda_1 = eigenvalues.max()
    return float(lambda_1 / total)


# ── main loop ────────────────────────────────────────────────────────────

def compute_egs_signals(mode: str = "benign") -> list:
    """
    Iterate all daily .npy files for a baseline in date order.
    For each consecutive pair compute ΔP_t, ΔD_t, ΔA_t.

    Parameters
    ----------
    mode : "benign" or "adv"

    Returns
    -------
    List of dicts with keys:
        date, n_tx, dispersion, anisotropy, delta_P, delta_D, delta_A
    """
    embed_dir = os.path.join(EMBED_DIR, mode)
    npy_files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in {embed_dir}\n"
            f"Run embed_snapshots.py first."
        )

    print(f"\n{'='*65}")
    print(f"  EGS — ΔP + ΔD + ΔA — {mode.upper()}")
    print(f"  Found {len(npy_files)} daily embedding files")
    print(f"{'='*65}")

    records    = []
    prev_proto = None
    prev_disp  = None
    prev_aniso = None

    for npy_path in npy_files:
        date_str  = os.path.basename(npy_path).replace(".npy", "")
        meta_path = os.path.join(embed_dir, f"meta_{date_str}.pkl")

        emb = np.load(npy_path)              # (n_tx, 64)

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

        print(
            f"  {date_str} | n={n_tx:5d} | "
            f"A={aniso:.4f} | "
            f"ΔP={delta_P:.6f} | "
            f"ΔD={delta_D:.6f} | "
            f"ΔA={delta_A:.6f}"
        )

        prev_proto = proto
        prev_disp  = disp
        prev_aniso = aniso

    print(f"\n  Done. {len(records)} records computed.")
    return records