"""
project_graph.py
----------------
Collapses a heterogeneous daily snapshot (customers, merchants, devices,
locations, transactions) into a transaction-only projected graph.

Two transaction nodes are connected in the projection if they share
at least one non-transaction entity (customer, merchant, device, or location).

Example:
    T_1 and T_2 both connect to M_99 (same merchant)
    → T_1 -- T_2 in projected graph (edge_reason="merchant")

This projected graph is what we count motifs on.
"""

import networkx as nx


def project_to_transaction_graph(G_snapshot):
    """
    Takes a daily heterogeneous snapshot and returns a transaction-only
    projected graph.

    Parameters
    ----------
    G_snapshot : nx.Graph
        A daily snapshot with mixed node types (transaction, customer,
        merchant, device, location).

    Returns
    -------
    G_proj : nx.Graph
        Projected graph where:
        - Nodes = transaction nodes only
        - Node attributes: fraud label, amount, date preserved
        - Edges = shared entity between two transactions
        - Edge attribute: shared_entity_type (customer/merchant/device/location)
    """
    G_proj = nx.Graph()

    # --- Step 1: Add all transaction nodes with their attributes ---
    tx_nodes = [
        (node, data)
        for node, data in G_snapshot.nodes(data=True)
        if data.get("node_type") == "transaction"
    ]

    for node, data in tx_nodes:
        G_proj.add_node(node, **data)

    # --- Step 2: For each non-transaction (entity) node, find all
    #             transactions connected to it. Those transactions
    #             are linked in the projection. ---

    entity_nodes = [
        (node, data)
        for node, data in G_snapshot.nodes(data=True)
        if data.get("node_type") != "transaction"
    ]

    for entity_node, entity_data in entity_nodes:
        entity_type = entity_data.get("node_type")  # customer/merchant/device/location

        # All transaction neighbors of this entity
        tx_neighbors = [
            n for n in G_snapshot.neighbors(entity_node)
            if G_snapshot.nodes[n].get("node_type") == "transaction"
        ]

        # Connect every pair of transactions that share this entity
        # If an edge already exists (shared via multiple entities), keep it
        for i in range(len(tx_neighbors)):
            for j in range(i + 1, len(tx_neighbors)):
                t1 = tx_neighbors[i]
                t2 = tx_neighbors[j]

                if not G_proj.has_edge(t1, t2):
                    G_proj.add_edge(
                        t1, t2,
                        shared_entity=entity_node,
                        shared_entity_type=entity_type
                    )
                # If edge exists, we don't overwrite — first shared entity is recorded

    return G_proj


def get_projection_stats(G_proj):
    """
    Quick diagnostics for a projected graph.
    Returns a dict with node count, edge count, fraud node count.
    """
    total_nodes = G_proj.number_of_nodes()
    total_edges = G_proj.number_of_edges()
    fraud_nodes = sum(
        1 for _, d in G_proj.nodes(data=True)
        if d.get("fraud") == 1
    )
    legit_nodes = total_nodes - fraud_nodes

    return {
        "total_tx_nodes": total_nodes,
        "total_edges": total_edges,
        "fraud_nodes": fraud_nodes,
        "legit_nodes": legit_nodes
    }