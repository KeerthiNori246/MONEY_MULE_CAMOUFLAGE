"""
compute_hub_signals.py
-----------------------
Computes raw hub redistribution signals for a single timestep.

Given two consecutive heterogeneous snapshots (G_curr, G_prev), returns:
    ΔD_t  — L1 norm of degree distribution change (entropy shift)
    ΔC_t  — L2 norm of top-k centrality vector change (hierarchy shift)

NOTE: Operates on the FULL heterogeneous graph (all node types).
      Hubs here are entity nodes (merchants, devices, customers, locations)
      connected to many transactions — NOT the projected tx-tx graph.

Design mirrors count_motifs.py so hub_redistribution.py can call this
the same way temporal_motif_evolution.py calls count_all_motifs().
"""

import numpy as np
from collections import Counter

# Number of top-k nodes to track for centrality vector.
# Tune up for denser graphs, down for speed.
TOP_K = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _degree_distribution(G):
    """
    Returns a normalized degree distribution as a numpy array.

    The array is a probability vector over degree values:
        index i  →  P(node has degree i)

    Normalization by total nodes ensures graph-size changes across
    days don't distort the L1 difference.
    """
    if G.number_of_nodes() == 0:
        return np.array([1.0])  # degenerate: all mass at degree 0

    degree_counts = Counter(dict(G.degree()).values())
    max_degree = max(degree_counts.keys())

    dist = np.zeros(max_degree + 1)
    for deg, count in degree_counts.items():
        dist[deg] = count

    total = dist.sum()
    return dist / total   # normalize → probability vector


def _top_k_centrality_vector(G, k=TOP_K):
    """
    Returns a numpy array of degree centrality values for the top-k
    nodes (by degree), sorted by node ID for consistency across timesteps.

    Sorting by node ID (not by rank) is critical: if we sorted by rank,
    a node swapping position would look like a huge vector change even
    if its actual centrality barely moved. Sorting by node ID means the
    vector encodes "what is node X's centrality today" consistently.

    Only nodes present in BOTH snapshots contribute a meaningful delta —
    new/disappeared nodes produce zeros in the absent timestep, which
    is correct behavior (a node appearing with high centrality IS a
    structural event worth capturing).
    """
    if G.number_of_nodes() == 0:
        return np.zeros(k)

    # Degree centrality: deg(v) / (n-1)
    n = G.number_of_nodes()
    if n <= 1:
        return np.zeros(k)

    degree_dict = dict(G.degree())
    centrality = {node: deg / (n - 1) for node, deg in degree_dict.items()}

    # Pick top-k nodes by centrality
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:k]
    top_nodes_sorted = sorted(top_nodes)  # sort by node ID for consistency

    vec = np.array([centrality[node] for node in top_nodes_sorted])

    # Pad with zeros if fewer than k nodes exist
    if len(vec) < k:
        vec = np.concatenate([vec, np.zeros(k - len(vec))])

    return vec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hub_signals(G_curr, G_prev=None):
    """
    Compute ΔD_t and ΔC_t for a single timestep.

    Parameters
    ----------
    G_curr : nx.Graph
        The snapshot at time t (full heterogeneous graph).
    G_prev : nx.Graph or None
        The snapshot at time t-1. If None (first timestep),
        both deltas return 0.

    Returns
    -------
    dict with keys:
        delta_entropy     : float  — L1 norm of degree distribution change
        delta_centrality  : float  — L2 norm of top-k centrality change
    """
    if G_prev is None:
        return {"delta_entropy": 0.0, "delta_centrality": 0.0}

    # --- ΔD_t: degree distribution entropy shift ---
    dist_curr = _degree_distribution(G_curr)
    dist_prev = _degree_distribution(G_prev)

    # Align lengths (different max degrees across days)
    max_len = max(len(dist_curr), len(dist_prev))
    dist_curr = np.pad(dist_curr, (0, max_len - len(dist_curr)))
    dist_prev = np.pad(dist_prev, (0, max_len - len(dist_prev)))

    delta_entropy = float(np.linalg.norm(dist_curr - dist_prev, ord=1))

    # --- ΔC_t: top-k centrality vector shift ---
    vec_curr = _top_k_centrality_vector(G_curr, k=TOP_K)
    vec_prev = _top_k_centrality_vector(G_prev, k=TOP_K)

    delta_centrality = float(np.linalg.norm(vec_curr - vec_prev, ord=2))

    return {
        "delta_entropy":    delta_entropy,
        "delta_centrality": delta_centrality,
    }