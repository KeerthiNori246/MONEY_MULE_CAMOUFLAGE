"""
embed_snapshots_drift.py
------------------------
Runs the frozen MLP encoder on structure-drift snapshots.
Identical logic to embed_snapshots.py — only the input folder
and output folder names change.

Reads:
    graphs/baselines/*_struct_drift_{intensity}.gpickle

Outputs:
    graphs/embeddings/struct_drift_low/YYYY-MM-DD.npy
    graphs/embeddings/struct_drift_low/meta_YYYY-MM-DD.pkl
    graphs/embeddings/struct_drift_medium/...
    graphs/embeddings/struct_drift_high/...
"""

import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR = os.path.join(_PROJECT_ROOT, "graphs", "baselines")
MODEL_DIR    = os.path.join(_PROJECT_ROOT, "graphs", "model")
EMBED_DIR    = os.path.join(_PROJECT_ROOT, "graphs", "embeddings")

INTENSITIES = ["low", "medium", "high"]

for intensity in INTENSITIES:
    os.makedirs(os.path.join(EMBED_DIR, f"struct_drift_{intensity}"), exist_ok=True)


# ── model (must match train_gnn.py v7 exactly) ───────────────────────────

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(out_dim, 1)

    def encode(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

    def forward(self, x):
        emb    = self.encode(x)
        logits = self.classifier(emb).squeeze(-1)
        return logits, emb


def load_frozen_model():
    config_path  = os.path.join(MODEL_DIR, "model_config.pkl")
    weights_path = os.path.join(MODEL_DIR, "graphsage_weights.pt")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    model = MLPEncoder(
        in_dim=config["in_dim"],
        hidden_dim=config["hidden_dim"],
        out_dim=config["out_dim"],
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Frozen MLP loaded (in={config['in_dim']} hidden={config['hidden_dim']} out={config['out_dim']})")
    return model, config


# ── feature extraction (identical to embed_snapshots.py) ─────────────────

def extract_snapshot_features(G_snap):
    tx_nodes, tx_labels = [], []
    for node, data in G_snap.nodes(data=True):
        if data.get("node_type") == "transaction":
            tx_nodes.append(node)
            tx_labels.append(int(data.get("fraud", 0)))

    n = len(tx_nodes)
    if n == 0:
        return None, [], []

    entity_fraud_ratio = {}
    for node, data in G_snap.nodes(data=True):
        if data.get("node_type") != "transaction":
            tx_nbs = [nb for nb in G_snap.neighbors(node)
                      if G_snap.nodes[nb].get("node_type") == "transaction"]
            if not tx_nbs:
                entity_fraud_ratio[node] = 0.0
            else:
                nf = sum(1 for nb in tx_nbs if G_snap.nodes[nb].get("fraud", 0) == 1)
                entity_fraud_ratio[node] = nf / len(tx_nbs)

    amounts       = np.zeros(n, dtype=np.float32)
    degrees       = np.zeros(n, dtype=np.float32)
    merch_degrees = np.zeros(n, dtype=np.float32)
    cust_degrees  = np.zeros(n, dtype=np.float32)
    fraud_nb_ratio = np.zeros(n, dtype=np.float32)

    for i, node in enumerate(tx_nodes):
        data = G_snap.nodes[node]
        amounts[i] = float(data.get("amount", 0.0))
        degrees[i] = float(G_snap.degree(node))
        for nb in G_snap.neighbors(node):
            nb_type = G_snap.nodes[nb].get("node_type", "")
            if nb_type == "merchant":
                merch_degrees[i] = float(G_snap.degree(nb))
            elif nb_type == "customer":
                cust_degrees[i]  = float(G_snap.degree(nb))
        entity_nbs = [nb for nb in G_snap.neighbors(node)
                      if G_snap.nodes[nb].get("node_type") != "transaction"]
        if entity_nbs:
            fraud_nb_ratio[i] = float(np.mean(
                [entity_fraud_ratio.get(nb, 0.0) for nb in entity_nbs]
            ))

    def norm(arr):
        mu, si = arr.mean(), arr.std()
        return (arr - mu) / si if si > 0 else arr - mu

    log_deg = np.log1p(degrees)
    x = np.stack([
        norm(amounts), norm(degrees), norm(log_deg),
        norm(merch_degrees), norm(cust_degrees), fraud_nb_ratio,
    ], axis=1).astype(np.float32)

    return torch.tensor(x, dtype=torch.float), tx_nodes, tx_labels


def embed_snapshot(model, G_snap):
    x, tx_node_ids, tx_labels = extract_snapshot_features(G_snap)
    if x is None:
        return None, [], []
    with torch.no_grad():
        emb_matrix = model.encode(x).numpy()
    return emb_matrix, tx_node_ids, tx_labels


# ── process one intensity level ──────────────────────────────────────────

def embed_drift_baseline(model, intensity: str):
    print(f"\n{'='*55}")
    print(f"  Embedding struct_drift_{intensity}")
    print(f"{'='*55}")

    pattern = os.path.join(BASELINE_DIR, f"*_struct_drift_{intensity}.gpickle")
    paths   = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No struct_drift_{intensity} baselines found.\n"
            f"Run create_structure_drift.py first."
        )

    print(f"  Found {len(paths)} snapshots")
    out_dir = os.path.join(EMBED_DIR, f"struct_drift_{intensity}")
    saved, skipped = 0, 0

    for path in tqdm(paths, desc=f"Embedding [struct_drift_{intensity}]"):
        date_str = os.path.basename(path).replace(f"_struct_drift_{intensity}.gpickle", "")

        with open(path, "rb") as f:
            G_snap = pickle.load(f)

        emb_matrix, tx_node_ids, tx_labels = embed_snapshot(model, G_snap)

        if emb_matrix is None:
            print(f"  ⚠ Skipping {date_str} — no transaction nodes")
            skipped += 1
            continue

        np.save(os.path.join(out_dir, f"{date_str}.npy"), emb_matrix)

        n_fraud = sum(tx_labels)
        meta = {
            "node_ids": tx_node_ids,
            "labels":   tx_labels,
            "date":     date_str,
            "mode":     f"struct_drift_{intensity}",
            "n_tx":     len(tx_node_ids),
            "n_fraud":  n_fraud,
            "n_legit":  len(tx_labels) - n_fraud,
        }
        with open(os.path.join(out_dir, f"meta_{date_str}.pkl"), "wb") as f:
            pickle.dump(meta, f)

        saved += 1

    print(f"  Saved: {saved}  Skipped: {skipped}")
    print(f"  Output → {out_dir}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  EMBEDDING — STRUCTURE DRIFT BASELINES")
    print("=" * 55)

    model, config = load_frozen_model()

    for intensity in INTENSITIES:
        embed_drift_baseline(model, intensity)

    print("\n" + "=" * 55)
    print("  DONE — Next: run compute_egs_drift.py")
    print("=" * 55)


if __name__ == "__main__":
    main()