from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from itertools import combinations
import numpy as np
import json
from tqdm import tqdm

# --- 1️⃣ Connect to Neo4j ---
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# --- 2️⃣ Load embedding model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- 3️⃣ Define helper similarity functions ---
def cosine_similarity(vec1, vec2):
    return float(util.cos_sim(vec1, vec2).item())

def jaccard_similarity(dict1, dict2):
    if not dict1 or not dict2:
        return 0.0
    keys1, keys2 = set(dict1.keys()), set(dict2.keys())
    intersect = keys1 & keys2
    if not intersect:
        return 0.0
    matches = sum(1 for k in intersect if str(dict1[k]).lower() == str(dict2[k]).lower())
    return matches / len(keys1 | keys2)

def hybrid_dedup_score(node_a, node_b):
    fuzzy_score = fuzz.token_set_ratio(node_a["name"], node_b["name"]) / 100
    if fuzzy_score > 0.9:
        return 1.0

    emb1 = embedder.encode(node_a["name"], convert_to_tensor=True)
    emb2 = embedder.encode(node_b["name"], convert_to_tensor=True)
    emb_sim = cosine_similarity(emb1, emb2)

    # print("node_a:", node_a["name"], "node_b:", node_b["name"])
    # print("node_a:", node_a["properties"], "node_b:", node_b["properties"])
    # print(json.loads(node_a["properties"]))
    # print(json.loads(node_b["properties"]))   
    prop_sim = jaccard_similarity(json.loads(node_a["properties"]), json.loads(node_b["properties"]))
    # print("emb_sim:", emb_sim, "prop_sim:", prop_sim)   
    #prop_sim = jaccard_similarity(node_a.get("properties", {}), node_b.get("properties", {}))
    label_match = 1.0 if node_a.get("label") == node_b.get("label") else 0.0

    score = 0.6 * emb_sim + 0.25 * prop_sim + 0.15 * label_match
    return round(score, 3)

# --- 4️⃣ Pull all nodes from Neo4j ---
def fetch_all_nodes():
    with driver.session() as session:
        query = "MATCH (n) RETURN id(n) as id, n.name as name, n.label as label, n.properties as properties"
        return [record.data() for record in session.run(query)]

# --- 5️⃣ Compute deduplication candidates ---
def find_merge_candidates(threshold=0.75):
    nodes = fetch_all_nodes()
    candidates = []

    for node_a, node_b in tqdm(combinations(nodes, 2), total=len(nodes)*(len(nodes)-1)//2):
        score = hybrid_dedup_score(node_a, node_b)
        if score >= threshold:
            candidates.append({
                "node_a": node_a,
                "node_b": node_b,
                "score": score
            })
    return sorted(candidates, key=lambda x: x["score"], reverse=True)

# --- 6️⃣ Optional: Merge function (manual or auto) ---
def merge_nodes(node_a_id, node_b_id):
    with driver.session() as session:
        query = """
        MATCH (a), (b)
        WHERE id(a) = $id1 AND id(b) = $id2
        WITH a, b
        CALL apoc.refactor.mergeNodes([a, b], {properties: "combine", mergeRels: true}) YIELD node
        RETURN node
        """
        session.run(query, {"id1": node_a_id, "id2": node_b_id})

# --- 7️⃣ Run it ---
if __name__ == "__main__":
    candidates = find_merge_candidates(threshold=0.75)
    print("\n=== Possible Duplicate Pairs ===")
    for c in candidates[:10]:
        print(f"{c['node_a']['name']}  ↔  {c['node_b']['name']}  |  score: {c['score']}")

    # Optional auto-merge example
    # for c in candidates:
    #     merge_nodes(c["node_a"]["id"], c["node_b"]["id"])
