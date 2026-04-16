"""
Directional Distance Encoding (DDE)
Replicates Section 3.1 of the SubgraphRAG paper.

Given a set of topic entities Tq and a subgraph G:
1. Initialize one-hot encodings at topic entities
2. Propagate FORWARD through directed edges (L rounds)
3. Propagate REVERSE through directed edges (L rounds)
4. Concatenate all rounds → structural encoding per entity
5. Triple encoding = [s_head || s_tail]
"""

import numpy as np
from collections import defaultdict
from neo4j import GraphDatabase

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ── STEP 1: LOAD SUBGRAPH FROM NEO4J ─────────────────────────────────────────
def load_subgraph(topic_entity: str, hops: int = 2) -> dict:
    """
    Load a local subgraph centered around the topic entity.
    Returns adjacency lists for forward and reverse directions.
    """
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS toLower($entity)
               OR toLower(b.name) CONTAINS toLower($entity)
            RETURN a.name AS head, r.type AS rel, b.name AS tail
            LIMIT 300
        """, entity=topic_entity)
        triples = [(r["head"], r["rel"], r["tail"]) for r in result
                   if None not in (r["head"], r["rel"], r["tail"])]

    # Build adjacency
    forward  = defaultdict(list)  # head → [tail]
    backward = defaultdict(list)  # tail → [head]
    entities = set()

    for h, r, t in triples:
        forward[h].append(t)
        backward[t].append(h)
        entities.add(h)
        entities.add(t)

    return {
        "triples":  triples,
        "forward":  dict(forward),
        "backward": dict(backward),
        "entities": list(entities)
    }

# ── STEP 2: DDE PROPAGATION ───────────────────────────────────────────────────
def compute_dde(topic_entities: list, subgraph: dict, L: int = 3) -> dict:
    """
    Compute Directional Distance Encoding for all entities.

    Args:
        topic_entities: list of topic entity names
        subgraph: output of load_subgraph()
        L: number of propagation rounds

    Returns:
        dict mapping entity_name → DDE vector (length 2*L + 2)
    """
    entities  = subgraph["entities"]
    forward   = subgraph["forward"]
    backward  = subgraph["backward"]
    n         = len(entities)
    ent_idx   = {e: i for i, e in enumerate(entities)}

    # Initialize: s^(0) = 1 if entity is topic entity, else 0
    s_fwd = np.zeros(n)
    s_rev = np.zeros(n)
    for te in topic_entities:
        for e in entities:
            if te.lower() in e.lower():
                idx = ent_idx[e]
                s_fwd[idx] = 1.0
                s_rev[idx] = 1.0

    # Store all rounds
    fwd_rounds = [s_fwd.copy()]
    rev_rounds = [s_rev.copy()]

    # Forward propagation: s^(l+1)[e] = MEAN(s^(l)[e'] for e' in in-neighbors of e)
    for _ in range(L):
        new_s = np.zeros(n)
        for e, neighbors in forward.items():
            if e in ent_idx:
                e_idx = ent_idx[e]
                vals  = [s_fwd[ent_idx[nb]] for nb in neighbors if nb in ent_idx]
                if vals:
                    new_s[e_idx] = np.mean(vals)
        s_fwd = new_s
        fwd_rounds.append(s_fwd.copy())

    # Reverse propagation: through backward edges
    for _ in range(L):
        new_s = np.zeros(n)
        for e, neighbors in backward.items():
            if e in ent_idx:
                e_idx = ent_idx[e]
                vals  = [s_rev[ent_idx[nb]] for nb in neighbors if nb in ent_idx]
                if vals:
                    new_s[e_idx] = np.mean(vals)
        s_rev = new_s
        rev_rounds.append(s_rev.copy())

    # Concatenate all rounds → entity DDE vector
    entity_dde = {}
    for e, i in ent_idx.items():
        fwd_vec = np.array([r[i] for r in fwd_rounds])
        rev_vec = np.array([r[i] for r in rev_rounds])
        entity_dde[e] = np.concatenate([fwd_vec, rev_vec])

    return entity_dde

# ── STEP 3: TRIPLE DDE ENCODING ──────────────────────────────────────────────
def get_triple_dde(triples: list, entity_dde: dict) -> np.ndarray:
    """
    For each triple (h, r, t), compute zτ = [dde(h) || dde(t)]
    Returns array of shape (num_triples, 2 * dde_dim)
    """
    encodings = []
    zero = np.zeros(list(entity_dde.values())[0].shape[0]) if entity_dde else np.zeros(8)

    for h, r, t in triples:
        h_enc = entity_dde.get(h, zero)
        t_enc = entity_dde.get(t, zero)
        encodings.append(np.concatenate([h_enc, t_enc]))

    return np.array(encodings) if encodings else np.zeros((0, len(zero)*2))

# ── STEP 4: DDE-BASED TRIPLE SCORING ─────────────────────────────────────────
def score_triples_with_dde(topic_entities: list, subgraph: dict) -> list:
    """
    Score triples by their structural proximity to topic entities using DDE.
    Higher score = closer to topic entity = more relevant.

    Returns sorted list of (triple, score) tuples.
    """
    entity_dde  = compute_dde(topic_entities, subgraph)
    triples     = subgraph["triples"]
    triple_dde  = get_triple_dde(triples, entity_dde)

    # Score = sum of DDE values (higher = closer to topic entity)
    scores = triple_dde.sum(axis=1)

    # Sort by score descending
    ranked = sorted(zip(triples, scores), key=lambda x: x[1], reverse=True)
    return ranked

# ── FULL DDE PIPELINE ─────────────────────────────────────────────────────────
def retrieve_with_dde(topic_entity: str, top_k: int = 50) -> list:
    """
    Full DDE retrieval pipeline.
    Returns top-K triples ranked by structural proximity.
    """
    print(f"🔍 Loading subgraph for: {topic_entity}")
    subgraph = load_subgraph(topic_entity)
    print(f"📦 Subgraph: {len(subgraph['entities'])} entities, {len(subgraph['triples'])} triples")

    print(f"🧮 Computing DDE encodings...")
    ranked = score_triples_with_dde([topic_entity], subgraph)

    top_k_triples = ranked[:top_k]
    print(f"✅ Selected top-{top_k} triples by DDE score")

    return [{"text": f"({h}, {r}, {t})", "score": float(s)}
            for (h, r, t), s in top_k_triples]

# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    triples = retrieve_with_dde("Alzheimer disease", top_k=10)
    print("\n📋 Top-10 DDE-ranked triples:")
    for i, t in enumerate(triples):
        print(f"  {i+1}. [score={t['score']:.4f}] {t['text']}")