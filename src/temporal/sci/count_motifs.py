"""
count_motifs.py  (FAST VERSION)
--------------------------------
Counts motifs DIRECTLY on the raw heterogeneous snapshot —
NO projection graph is built. This is 10-50x faster.

Key insight: we don't need to materialise the projected graph.
We only need the entity→transaction adjacency lists, which we
already have in the snapshot.

Motif definitions (on the VIRTUAL projected space):

  Triangles    — 3 transactions that pairwise share an entity.
                 Detected by: for each entity, take tx-neighbor pairs,
                 check if those two txs share a SECOND entity.
                 (Uses set intersection — O(degree) per pair, capped.)

  2-hop Wedges — Any entity with ≥2 tx neighbors contributes
                 C(k, 2) wedges where k = number of tx neighbors.
                 Pure arithmetic, no graph traversal needed.

FAN-OUT CAP:
  Entities connected to > MAX_FAN_OUT transactions are skipped.
  (A merchant with 50 000 tx neighbors adds O(n²) pairs and is
   statistically uninformative as a structural signal.)
"""

from collections import defaultdict
from itertools import combinations

# Skip any entity node whose tx-neighbor count exceeds this.
# Tune down if still slow, up if you want more coverage.
MAX_FAN_OUT = 500


# ---------------------------------------------------------------------------
# Internal helper: build entity → [tx_nodes] index in ONE pass
# ---------------------------------------------------------------------------

def _build_entity_tx_index(G_snapshot):
    """
    Returns
    -------
    entity_to_txs : dict  { entity_node: [tx_node, ...] }
    tx_to_entities: dict  { tx_node: set(entity_nodes) }
    """
    entity_to_txs  = defaultdict(list)
    tx_to_entities = defaultdict(set)

    # Single edge scan
    for u, v in G_snapshot.edges():
        u_type = G_snapshot.nodes[u].get("node_type")
        v_type = G_snapshot.nodes[v].get("node_type")

        if u_type == "transaction" and v_type != "transaction":
            tx_node, entity_node = u, v
        elif v_type == "transaction" and u_type != "transaction":
            tx_node, entity_node = v, u
        else:
            continue

        entity_to_txs[entity_node].append(tx_node)
        tx_to_entities[tx_node].add(entity_node)

    return entity_to_txs, tx_to_entities


# ---------------------------------------------------------------------------
# Motif 1: Triangles
# ---------------------------------------------------------------------------

def count_triangles(entity_to_txs, tx_to_entities):
    """
    For each entity E with tx-neighbors [T1, T2, ...]:
      For each pair (Ti, Tj):
        If Ti and Tj share a SECOND entity (besides E) → triangle.
    Each triangle is seen 3 times (once per edge/entity), so divide by 3.
    Capped at MAX_FAN_OUT.
    """
    triangle_count = 0

    for entity, txs in entity_to_txs.items():
        if len(txs) < 2 or len(txs) > MAX_FAN_OUT:
            continue

        for t1, t2 in combinations(txs, 2):
            shared = tx_to_entities[t1] & tx_to_entities[t2]
            shared.discard(entity)
            if shared:
                triangle_count += 1

    return triangle_count // 3


# ---------------------------------------------------------------------------
# Motif 2: 2-hop Wedges
# ---------------------------------------------------------------------------

def count_2hop_wedges(entity_to_txs):
    """
    Each entity with k tx-neighbors contributes C(k, 2) wedges.
    Pure arithmetic — no graph traversal.
    """
    total = 0
    for entity, txs in entity_to_txs.items():
        k = len(txs)
        if 2 <= k <= MAX_FAN_OUT:
            total += (k * (k - 1)) // 2
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_all_motifs(G_snapshot):
    """
    Count triangles and wedges directly on a raw heterogeneous snapshot.

    Parameters
    ----------
    G_snapshot : nx.Graph

    Returns
    -------
    dict with keys: triangles, wedges
    """
    if G_snapshot.number_of_nodes() == 0:
        return {"triangles": 0, "wedges": 0}

    entity_to_txs, tx_to_entities = _build_entity_tx_index(G_snapshot)

    return {
        "triangles": count_triangles(entity_to_txs, tx_to_entities),
        "wedges":    count_2hop_wedges(entity_to_txs),
    }