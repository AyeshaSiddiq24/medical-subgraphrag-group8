"""
MLP Triple Retriever with DDE
Replicates Section 3.1 of SubgraphRAG paper.
Fixed weak supervision: positive = indication only
"""

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from dde import load_subgraph, compute_dde

EMBED_DIM = 384
DDE_DIM   = 8
INPUT_DIM = EMBED_DIM * 4 + DDE_DIM * 2

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── MLP ───────────────────────────────────────────────────────────────────────
class TripleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128),       nn.ReLU(),
            nn.Linear(128, 1),         nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── WEAK SUPERVISION ──────────────────────────────────────────────────────────
def generate_training_data(subgraph, entity_dde):
    """
    Positive = indication triples (drug treats disease)
    Negative = all other relation types
    """
    triples  = subgraph["triples"]
    zero_dde = np.zeros(DDE_DIM)
    X, y     = [], []

    pos_count = 0
    neg_count = 0

    for h, r, t in triples:
        # ONLY indication = positive
        if r == "indication":
            label = 1.0
            pos_count += 1
        else:
            label = 0.0
            neg_count += 1

        h_enc = embedder.encode(h, convert_to_numpy=True)
        r_enc = embedder.encode(r, convert_to_numpy=True)
        t_enc = embedder.encode(t, convert_to_numpy=True)
        dde_h = entity_dde.get(h, zero_dde)[:DDE_DIM]
        dde_t = entity_dde.get(t, zero_dde)[:DDE_DIM]
        z_tau = np.concatenate([dde_h, dde_t])
        feat  = np.concatenate([np.zeros(EMBED_DIM), h_enc, r_enc, t_enc, z_tau])
        X.append(feat)
        y.append(label)

    print(f"  Positives (indication): {pos_count} | Negatives: {neg_count}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train_mlp(X, y, epochs=50, lr=1e-3):
    model     = TripleMLP(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights to handle imbalance
    pos     = y.sum()
    neg     = len(y) - pos
    pos_w   = torch.tensor(neg / (pos + 1e-10), dtype=torch.float32)
    criterion = nn.BCELoss(reduction='none')

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds  = model(X_t)
        losses = criterion(preds, y_t)
        # Weight positive samples more
        weights = torch.where(y_t == 1, pos_w, torch.ones_like(y_t))
        loss    = (losses * weights).mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

    return model

# ── MLP SCORING ───────────────────────────────────────────────────────────────
def score_with_mlp(query, triples, model, entity_dde):
    zero_dde = np.zeros(DDE_DIM)
    q_enc    = embedder.encode(query, convert_to_numpy=True)
    feats    = []
    for h, r, t in triples:
        h_enc = embedder.encode(h, convert_to_numpy=True)
        r_enc = embedder.encode(r, convert_to_numpy=True)
        t_enc = embedder.encode(t, convert_to_numpy=True)
        dde_h = entity_dde.get(h, zero_dde)[:DDE_DIM]
        dde_t = entity_dde.get(t, zero_dde)[:DDE_DIM]
        z_tau = np.concatenate([dde_h, dde_t])
        feat  = np.concatenate([q_enc, h_enc, r_enc, t_enc, z_tau])
        feats.append(feat)
    X_t   = torch.tensor(np.array(feats, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        scores = model(X_t).numpy()
    return sorted(zip(triples, scores), key=lambda x: x[1], reverse=True)

# ── COSINE BASELINE ───────────────────────────────────────────────────────────
def score_with_cosine(query, triples):
    q_enc  = embedder.encode(query, convert_to_numpy=True)
    q_enc  = q_enc / (np.linalg.norm(q_enc) + 1e-10)
    texts  = [f"{h} {r} {t}" for h, r, t in triples]
    t_encs = embedder.encode(texts, convert_to_numpy=True, batch_size=64)
    t_encs = t_encs / (np.linalg.norm(t_encs, axis=1, keepdims=True) + 1e-10)
    scores = t_encs @ q_enc
    return sorted(zip(triples, scores), key=lambda x: x[1], reverse=True)

# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
def retrieve_with_mlp_dde(query, topic_entity, top_k=50):
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"📍 Topic: {topic_entity}")
    print('='*60)

    subgraph = load_subgraph(topic_entity)
    triples  = subgraph["triples"]
    print(f"📦 {len(triples)} triples, {len(subgraph['entities'])} entities")

    print("🧮 Computing DDE encodings...")
    entity_dde = compute_dde([topic_entity], subgraph)

    print("🏋️ Generating weak supervision training data...")
    X, y  = generate_training_data(subgraph, entity_dde)

    if y.sum() == 0:
        print("⚠️  No indication triples found — falling back to cosine")
        cosine_ranked = score_with_cosine(query, triples)
        cosine_top    = cosine_ranked[:top_k]
        return {
            "mlp_dde": [{"text": f"({h},{r},{t})", "score": float(s)} for (h,r,t),s in cosine_top],
            "cosine":  [{"text": f"({h},{r},{t})", "score": float(s)} for (h,r,t),s in cosine_top],
        }

    print("🤖 Training MLP (50 epochs)...")
    model = train_mlp(X, y, epochs=50)

    print("📊 Scoring with MLP+DDE...")
    mlp_ranked = score_with_mlp(query, triples, model, entity_dde)
    mlp_top    = mlp_ranked[:top_k]

    print("📊 Scoring with cosine similarity...")
    cosine_ranked = score_with_cosine(query, triples)
    cosine_top    = cosine_ranked[:top_k]

    print(f"\n{'─'*60}")
    print("🏆 MLP+DDE Top-5:")
    for i, ((h,r,t), s) in enumerate(mlp_top[:5]):
        print(f"  {i+1}. [{s:.3f}] ({h}, {r}, {t})")

    print(f"\n📉 Cosine Baseline Top-5:")
    for i, ((h,r,t), s) in enumerate(cosine_top[:5]):
        print(f"  {i+1}. [{s:.3f}] ({h}, {r}, {t})")

    mlp_ind    = sum(1 for (h,r,t),_ in mlp_top[:10] if r == "indication")
    cosine_ind = sum(1 for (h,r,t),_ in cosine_top[:10] if r == "indication")
    print(f"\n✅ Indication triples in top-10 — MLP+DDE: {mlp_ind} | Cosine: {cosine_ind}")

    return {
        "mlp_dde": [{"text": f"({h}, {r}, {t})", "score": float(s)} for (h,r,t),s in mlp_top],
        "cosine":  [{"text": f"({h}, {r}, {t})", "score": float(s)} for (h,r,t),s in cosine_top],
        "model":   model,
        "entity_dde": entity_dde,
        "mlp_indication_count":    mlp_ind,
        "cosine_indication_count": cosine_ind,
    }

# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = retrieve_with_mlp_dde(
        query        = "What drugs treat Alzheimer disease?",
        topic_entity = "Alzheimer disease",
        top_k        = 10
    )