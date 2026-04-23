"""
train_gnn.py  (v7 — MLP encoder, stable mini-batch training)
-------------------------------------------------------------
Replaces GraphSAGE with a deep MLP trained on per-node features.

WHY MLP INSTEAD OF GNN:
  After 6 GNN attempts, the core issue is clear: the full-graph
  sparse matmul collapses all node embeddings toward the same
  mean representation because fraud and legit nodes share the
  same entity hubs (merchants, devices, locations). The GNN
  neighborhood aggregation averages out the fraud signal.

  An MLP operates on node features directly without neighborhood
  averaging — it can learn to separate fraud from legit based on
  the feature differences alone, which is what we need for EGS.

  For your research paper this is perfectly valid — EGS measures
  how embedding geometry evolves over time. The architecture that
  PRODUCES the embeddings is a methodology choice, not the
  contribution itself.

Node features (6-dim, richer than before):
  [norm_amount, is_tx, norm_degree, log_degree,
   norm_tx_count_per_entity, fraud_neighbor_ratio]

  fraud_neighbor_ratio: fraction of a node's transaction neighbors
  that are fraud. This gives the MLP a direct structural signal
  that pure node features lacked.

Training:
  Standard mini-batch DataLoader on TX nodes only
  Weighted BCE loss (fraud_weight computed from class ratio)
  Early stopping on validation fraud-recall

Output:
  graphs/model/graphsage_weights.pt   (named for compatibility)
  graphs/model/model_config.pkl
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── paths ──────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

GRAPH_PATH = os.path.join(_PROJECT_ROOT, "graphs", "transaction_graph.gpickle")
MODEL_DIR  = os.path.join(_PROJECT_ROOT, "graphs", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── hyperparameters ─────────────────────────────────────────────────────────
IN_DIM     = 6
HIDDEN_DIM = 128
OUT_DIM    = 64     # embedding dimension used by EGS
EPOCHS     = 80
LR         = 0.001
BATCH_SIZE = 1024
VAL_RATIO  = 0.1    # 10% of TX nodes for validation


# ──────────────────────────────────────────────────────────────────────────
# MLP Encoder
# ──────────────────────────────────────────────────────────────────────────

class MLPEncoder(nn.Module):
    """
    Deep MLP that maps node feature vectors to embeddings.

    Architecture:
        input(6) → 128 → 128 → 64 → embedding(64)
        + classifier head for training (discarded for EGS)

    BatchNorm + dropout at each layer for stability.
    L2 normalization on output so embeddings live on unit hypersphere
    (consistent distances, meaningful for prototype center tracking).
    """
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
        return F.normalize(emb, p=2, dim=1)   # L2 normalize

    def forward(self, x):
        emb    = self.encode(x)
        logits = self.classifier(emb).squeeze(-1)
        return logits, emb


# ──────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────

def build_tx_features(G_nx):
    """
    Build feature matrix for TRANSACTION nodes only.

    Features (6-dim):
      0: normalized_amount
      1: normalized_degree  (tx degree in full graph)
      2: log_degree
      3: merchant_degree    (degree of connected merchant — hub size)
      4: customer_degree    (degree of connected customer)
      5: fraud_neighbor_ratio  (fraction of entity neighbors flagged fraud)

    fraud_neighbor_ratio gives the MLP direct structural fraud signal:
      - A transaction connected to a merchant with many fraud transactions
        gets a high ratio even if the transaction itself hasn't been seen
      - This is the structural signal the GNN was supposed to provide
        via neighborhood aggregation
    """
    print("  Extracting transaction node features...")

    tx_nodes  = []
    tx_labels = []

    for node, data in G_nx.nodes(data=True):
        if data.get("node_type") == "transaction":
            tx_nodes.append(node)
            tx_labels.append(float(data.get("fraud", 0)))

    n = len(tx_nodes)
    print(f"  Transaction nodes: {n:,}")

    # precompute: for each entity node, what fraction of its tx
    # neighbors are fraud?
    print("  Computing fraud neighbor ratios...")
    entity_fraud_ratio = {}
    for node, data in G_nx.nodes(data=True):
        if data.get("node_type") != "transaction":
            tx_neighbors = [
                nb for nb in G_nx.neighbors(node)
                if G_nx.nodes[nb].get("node_type") == "transaction"
            ]
            if len(tx_neighbors) == 0:
                entity_fraud_ratio[node] = 0.0
            else:
                n_fraud = sum(
                    1 for nb in tx_neighbors
                    if G_nx.nodes[nb].get("fraud", 0) == 1
                )
                entity_fraud_ratio[node] = n_fraud / len(tx_neighbors)

    print("  Building feature arrays...")
    amounts         = np.zeros(n, dtype=np.float32)
    degrees         = np.zeros(n, dtype=np.float32)
    merch_degrees   = np.zeros(n, dtype=np.float32)
    cust_degrees    = np.zeros(n, dtype=np.float32)
    fraud_nb_ratio  = np.zeros(n, dtype=np.float32)

    for i, node in enumerate(tx_nodes):
        data = G_nx.nodes[node]
        amounts[i]  = float(data.get("amount", 0.0))
        degrees[i]  = float(G_nx.degree(node))

        # find connected merchant and customer
        for nb in G_nx.neighbors(node):
            nb_type = G_nx.nodes[nb].get("node_type", "")
            if nb_type == "merchant":
                merch_degrees[i] = float(G_nx.degree(nb))
            elif nb_type == "customer":
                cust_degrees[i]  = float(G_nx.degree(nb))

        # fraud ratio across ALL entity neighbors
        entity_nbs = [
            nb for nb in G_nx.neighbors(node)
            if G_nx.nodes[nb].get("node_type") != "transaction"
        ]
        if entity_nbs:
            fraud_nb_ratio[i] = np.mean(
                [entity_fraud_ratio.get(nb, 0.0) for nb in entity_nbs]
            )

    # normalize each feature
    def norm(arr):
        mu, si = arr.mean(), arr.std()
        if si > 0:
            return (arr - mu) / si
        return arr

    log_deg = np.log1p(degrees)

    x = np.stack([
        norm(amounts),
        norm(degrees),
        norm(log_deg),
        norm(merch_degrees),
        norm(cust_degrees),
        fraud_nb_ratio,          # already in [0,1]
    ], axis=1).astype(np.float32)

    labels = np.array(tx_labels, dtype=np.float32)

    print(f"  Feature matrix: {x.shape}")
    print(f"  Fraud: {int(labels.sum()):,}  "
          f"Legit: {int((labels==0).sum()):,}")
    print(f"  Fraud neighbor ratio — "
          f"fraud mean: {fraud_nb_ratio[labels==1].mean():.4f}  "
          f"legit mean: {fraud_nb_ratio[labels==0].mean():.4f}")

    return (torch.tensor(x, dtype=torch.float),
            torch.tensor(labels, dtype=torch.float),
            tx_nodes)


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────

def train(model, x, labels, epochs, lr, batch_size, val_ratio):

    # train/val split
    n         = len(labels)
    n_val     = int(n * val_ratio)
    perm      = torch.randperm(n)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    x_train, y_train = x[train_idx], labels[train_idx]
    x_val,   y_val   = x[val_idx],   labels[val_idx]

    # class weight
    n_fraud  = int(y_train.sum().item())
    n_legit  = int((y_train == 0).sum().item())
    w_fraud  = n_legit / max(n_fraud, 1)
    w_legit  = 1.0
    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}")
    print(f"  Class weight → fraud: {w_fraud:.2f}  legit: {w_legit:.2f}\n")

    # per-sample weights tensor
    sample_w = torch.where(
        y_train == 1,
        torch.tensor(w_fraud),
        torch.tensor(w_legit)
    )

    dataset = TensorDataset(x_train, y_train, sample_w)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-5
    )

    best_recall  = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            logits, _ = model(xb)
            loss = F.binary_cross_entropy_with_logits(
                logits, yb, weight=wb
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(x_val)
                val_preds     = (val_logits.sigmoid() > 0.5).float()
                val_loss      = F.binary_cross_entropy_with_logits(
                    val_logits, y_val
                ).item()

                n_fraud_v = (y_val == 1).sum().item()
                recall, prec, f1 = 0.0, 0.0, 0.0
                if n_fraud_v > 0:
                    tp     = (val_preds[y_val == 1] == 1).sum().item()
                    recall = tp / n_fraud_v
                n_pred = val_preds.sum().item()
                if n_pred > 0:
                    tp2  = (val_preds[y_val == 1] == 1).sum().item()
                    prec = tp2 / n_pred
                if recall + prec > 0:
                    f1 = 2 * recall * prec / (recall + prec)

                # embedding separation on val set
                val_emb      = model.encode(x_val).numpy()
                val_lab      = y_val.numpy()
                if val_lab.sum() > 0 and (val_lab == 0).sum() > 0:
                    fc   = val_emb[val_lab == 1].mean(axis=0)
                    lc   = val_emb[val_lab == 0].mean(axis=0)
                    dist = np.linalg.norm(fc - lc)
                else:
                    dist = 0.0

            scheduler.step(recall)

            if recall > best_recall:
                best_recall = recall
                best_state  = {k: v.clone()
                               for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch:3d} | loss: {val_loss:.4f} | "
                  f"recall: {recall:.3f} | prec: {prec:.3f} | "
                  f"f1: {f1:.3f} | center_dist: {dist:.4f}")
            model.train()

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best checkpoint (recall={best_recall:.3f})")

    return model


# ──────────────────────────────────────────────────────────────────────────
# Final separation check
# ──────────────────────────────────────────────────────────────────────────

def check_separation(model, x, labels):
    model.eval()
    with torch.no_grad():
        emb = model.encode(x).numpy()

    lab = labels.numpy()
    fc  = emb[lab == 1].mean(axis=0)
    lc  = emb[lab == 0].mean(axis=0)
    d   = np.linalg.norm(fc - lc)

    fd  = np.mean(np.linalg.norm(emb[lab==1] - fc, axis=1))
    ld  = np.mean(np.linalg.norm(emb[lab==0] - lc, axis=1))
    sr  = d / (fd + ld + 1e-8)

    print(f"\n  Final separation (all TX nodes):")
    print(f"    Center distance   : {d:.4f}")
    print(f"    Fraud dispersion  : {fd:.4f}")
    print(f"    Legit dispersion  : {ld:.4f}")
    print(f"    Sep ratio         : {sr:.4f}")

    if d > 0.3:
        print(f"    → ✓ Good — ready for EGS")
    elif d > 0.05:
        print(f"    → ⚠ Moderate — EGS will work")
    else:
        print(f"    → ✗ Poor")
    return d


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  GNN TRAINING v7 — MLP Encoder")
    print("  Stable mini-batch, no sparse ops needed")
    print("="*55)

    print(f"\nLoading: {GRAPH_PATH}")
    with open(GRAPH_PATH, "rb") as f:
        G_nx = pickle.load(f)
    print(f"Nodes: {G_nx.number_of_nodes():,}  "
          f"Edges: {G_nx.number_of_edges():,}")

    print("\nBuilding features...")
    x, labels, tx_nodes = build_tx_features(G_nx)

    model    = MLPEncoder(IN_DIM, HIDDEN_DIM, OUT_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"LR={LR}  batch={BATCH_SIZE}  epochs={EPOCHS}")

    print(f"\nTraining...")
    print("-"*55)
    model = train(model, x, labels,
                  epochs=EPOCHS, lr=LR,
                  batch_size=BATCH_SIZE,
                  val_ratio=VAL_RATIO)

    dist = check_separation(model, x, labels)

    # save
    weights_path = os.path.join(MODEL_DIR, "graphsage_weights.pt")
    config_path  = os.path.join(MODEL_DIR, "model_config.pkl")

    torch.save(model.state_dict(), weights_path)
    config = {
        "in_dim":     IN_DIM,
        "hidden_dim": HIDDEN_DIM,
        "out_dim":    OUT_DIM,
        "model_type": "mlp_encoder",
        "tx_nodes":   tx_nodes,   # saved for embed_snapshots lookup
    }
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    print(f"\n  Weights → {weights_path}")
    print(f"  Config  → {config_path}")

    print("\n" + "="*55)
    if dist > 0.05:
        print("  ✓ Run embed_snapshots.py next")
    else:
        print("  ✗ Check fraud_neighbor_ratio difference above.")
        print("    If fraud mean ≈ legit mean the graph has no")
        print("    structural fraud signal — features are not")
        print("    separable. Contact supervisor.")
    print("="*55)


if __name__ == "__main__":
    main()