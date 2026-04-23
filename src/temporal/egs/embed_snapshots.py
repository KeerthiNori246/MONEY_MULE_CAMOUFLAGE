"""
embed_snapshots.py  (v5 — MLP encoder, matches train_gnn v7)
-------------------------------------------------------------
Loads frozen MLPEncoder and runs inference on every daily
baseline snapshot to produce per-day embedding matrices.

Key difference from GNN versions:
  The MLP operates on per-node features directly — no adjacency
  matrix needed at inference time. Each transaction node is
  embedded independently based on its 6 features.

  This means embed_snapshot is just:
      features = extract_tx_features(snapshot)
      embeddings = model.encode(features)

Output per baseline per day:
  graphs/embeddings/benign/YYYY-MM-DD.npy    (n_tx, 64)
  graphs/embeddings/benign/meta_YYYY-MM-DD.pkl
  graphs/embeddings/adv/YYYY-MM-DD.npy
  graphs/embeddings/adv/meta_YYYY-MM-DD.pkl
"""

import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

BASELINE_DIR = os.path.join(_PROJECT_ROOT, "graphs", "baselines")
MODEL_DIR    = os.path.join(_PROJECT_ROOT, "graphs", "model")
EMBED_DIR    = os.path.join(_PROJECT_ROOT, "graphs", "embeddings")

for sub in ["benign", "adv"]:
    os.makedirs(os.path.join(EMBED_DIR, sub), exist_ok=True)

print(f"[paths] BASELINE_DIR = {BASELINE_DIR}")
print(f"[paths] MODEL_DIR    = {MODEL_DIR}")
print(f"[paths] EMBED_DIR    = {EMBED_DIR}")


# ──────────────────────────────────────────────────────────────────────────
# Model — must exactly match train_gnn.py v7
# ──────────────────────────────────────────────────────────────────────────

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
        emb = self.encoder(x)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x):
        emb    = self.encode(x)
        logits = self.classifier(emb).squeeze(-1)
        return logits, emb


# ──────────────────────────────────────────────────────────────────────────
# Load frozen model
# ──────────────────────────────────────────────────────────────────────────

def load_frozen_model():
    config_path  = os.path.join(MODEL_DIR, "model_config.pkl")
    weights_path = os.path.join(MODEL_DIR, "graphsage_weights.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\nRun train_gnn.py first."
        )

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    if config.get("model_type") != "mlp_encoder":
        raise ValueError(
            "Config is not from train_gnn v7 (mlp_encoder). "
            "Re-run train_gnn.py first."
        )

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

    print(f"Frozen MLP encoder loaded  "
          f"(in={config['in_dim']} hidden={config['hidden_dim']} "
          f"out={config['out_dim']})")
    return model, config


# ──────────────────────────────────────────────────────────────────────────
# Feature extraction for a single snapshot
# Must match train_gnn.py v7 feature construction EXACTLY
# ──────────────────────────────────────────────────────────────────────────

def extract_snapshot_features(G_snap):
    """
    Extract 6-dim features for every transaction node in a snapshot.

    Features (same order as train_gnn v7):
      0: norm_amount
      1: norm_degree
      2: norm_log_degree
      3: norm_merchant_degree
      4: norm_customer_degree
      5: fraud_neighbor_ratio

    Returns
    -------
    x           : (n_tx, 6) float tensor
    tx_node_ids : list of tx node IDs
    tx_labels   : list of fraud labels (0/1)
    """
    tx_nodes  = []
    tx_labels = []

    for node, data in G_snap.nodes(data=True):
        if data.get("node_type") == "transaction":
            tx_nodes.append(node)
            tx_labels.append(int(data.get("fraud", 0)))

    n = len(tx_nodes)
    if n == 0:
        return None, [], []

    # entity fraud ratios within this snapshot
    entity_fraud_ratio = {}
    for node, data in G_snap.nodes(data=True):
        if data.get("node_type") != "transaction":
            tx_nbs = [
                nb for nb in G_snap.neighbors(node)
                if G_snap.nodes[nb].get("node_type") == "transaction"
            ]
            if not tx_nbs:
                entity_fraud_ratio[node] = 0.0
            else:
                nf = sum(
                    1 for nb in tx_nbs
                    if G_snap.nodes[nb].get("fraud", 0) == 1
                )
                entity_fraud_ratio[node] = nf / len(tx_nbs)

    amounts        = np.zeros(n, dtype=np.float32)
    degrees        = np.zeros(n, dtype=np.float32)
    merch_degrees  = np.zeros(n, dtype=np.float32)
    cust_degrees   = np.zeros(n, dtype=np.float32)
    fraud_nb_ratio = np.zeros(n, dtype=np.float32)

    for i, node in enumerate(tx_nodes):
        data = G_snap.nodes[node]
        amounts[i]  = float(data.get("amount", 0.0))
        degrees[i]  = float(G_snap.degree(node))

        for nb in G_snap.neighbors(node):
            nb_type = G_snap.nodes[nb].get("node_type", "")
            if nb_type == "merchant":
                merch_degrees[i] = float(G_snap.degree(nb))
            elif nb_type == "customer":
                cust_degrees[i]  = float(G_snap.degree(nb))

        entity_nbs = [
            nb for nb in G_snap.neighbors(node)
            if G_snap.nodes[nb].get("node_type") != "transaction"
        ]
        if entity_nbs:
            fraud_nb_ratio[i] = float(np.mean(
                [entity_fraud_ratio.get(nb, 0.0) for nb in entity_nbs]
            ))

    def norm(arr):
        mu, si = arr.mean(), arr.std()
        if si > 0:
            return (arr - mu) / si
        return arr - mu

    log_deg = np.log1p(degrees)

    x = np.stack([
        norm(amounts),
        norm(degrees),
        norm(log_deg),
        norm(merch_degrees),
        norm(cust_degrees),
        fraud_nb_ratio,
    ], axis=1).astype(np.float32)

    return torch.tensor(x, dtype=torch.float), tx_nodes, tx_labels


# ──────────────────────────────────────────────────────────────────────────
# Embed one snapshot
# ──────────────────────────────────────────────────────────────────────────

def embed_snapshot(model, G_snap):
    x, tx_node_ids, tx_labels = extract_snapshot_features(G_snap)

    if x is None:
        return None, [], []

    with torch.no_grad():
        emb_matrix = model.encode(x).numpy()   # (n_tx, 64)

    return emb_matrix, tx_node_ids, tx_labels


# ──────────────────────────────────────────────────────────────────────────
# Process all snapshots for one baseline
# ──────────────────────────────────────────────────────────────────────────

def embed_baseline(model, mode="benign"):
    print(f"\n{'='*50}")
    print(f"Embedding snapshots — {mode.upper()}")
    print(f"{'='*50}")

    pattern = os.path.join(BASELINE_DIR, f"*_{mode}.gpickle")
    paths   = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No {mode} baselines in {BASELINE_DIR}.")

    print(f"Found {len(paths)} snapshots")
    out_dir = os.path.join(EMBED_DIR, mode)
    saved, skipped = 0, 0

    for path in tqdm(paths, desc=f"Embedding [{mode}]"):
        date_str = os.path.basename(path).replace(f"_{mode}.gpickle", "")

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
            "mode":     mode,
            "n_tx":     len(tx_node_ids),
            "n_fraud":  n_fraud,
            "n_legit":  len(tx_labels) - n_fraud,
        }
        with open(os.path.join(out_dir, f"meta_{date_str}.pkl"), "wb") as f:
            pickle.dump(meta, f)

        saved += 1

    print(f"\nSaved: {saved}  Skipped: {skipped}")
    print(f"Output → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────────────

def sanity_check(mode="benign"):
    """
    Load first saved embedding and print basic stats.
    For EGS we only need embeddings to be non-trivial —
    the separation is checked on the full graph in train_gnn.py.
    """
    out_dir   = os.path.join(EMBED_DIR, mode)
    npy_files = sorted(glob.glob(os.path.join(out_dir, "*.npy")))

    if not npy_files:
        print(f"  No .npy files for {mode}")
        return

    # pick a day that has both fraud and legit if possible
    for npy_path in npy_files:
        date_str  = os.path.basename(npy_path).replace(".npy", "")
        meta_path = os.path.join(out_dir, f"meta_{date_str}.pkl")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        emb    = np.load(npy_path)
        labels = np.array(meta["labels"])

        print(f"\n  [{mode}] {date_str}:")
        print(f"    Shape      : {emb.shape}")
        print(f"    Fraud nodes: {meta['n_fraud']}")
        print(f"    Legit nodes: {meta['n_legit']}")
        print(f"    Emb mean   : {emb.mean():.4f}")
        print(f"    Emb std    : {emb.std():.4f}")
        print(f"    Emb norm   : {np.linalg.norm(emb[0]):.4f} "
              f"(should be ≈ 1.0 — L2 normalized)")
        break


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  EMBEDDING SNAPSHOTS v5 — MLP Encoder")
    print("="*55)

    model, config = load_frozen_model()

    embed_baseline(model, mode="benign")
    embed_baseline(model, mode="adv")

    print("\n" + "-"*55)
    print("SANITY CHECKS")
    sanity_check("benign")
    sanity_check("adv")

    print("\n" + "="*55)
    print("  DONE")
    print(f"  Embeddings → {EMBED_DIR}")
    print("  Next: compute_egs_signals.py")
    print("="*55)


if __name__ == "__main__":
    main()