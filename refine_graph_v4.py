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

# Load a strong multilingual model (good for abbreviations too)
embedder = SentenceTransformer("all-MiniLM-L12-v2")

def cosine_similarity(vec1, vec2):
    return float(util.cos_sim(vec1, vec2))

def compute_property_similarity(props1, props2):
    """Compute Jaccard similarity over property keys and values"""
    if not props1 or not props2:
        return 0.0
    shared_keys = set(props1.keys()) & set(props2.keys())
    if not shared_keys:
        return 0.0
    
    matches = sum(1 for k in shared_keys if props1[k] == props2[k])
    return matches / len(shared_keys)

def compute_neighbor_overlap(node1_name, node2_name, label1, label2):
    """Compute overlap of neighbors in Neo4j"""
    with driver.session() as session:
        query = """
        MATCH (n1:{l1} {{name: $n1}})-[]->(nbr1)
        MATCH (n2:{l2} {{name: $n2}})-[]->(nbr2)
        RETURN apoc.coll.intersection(collect(DISTINCT nbr1.name), collect(DISTINCT nbr2.name)) AS shared,
            collect(DISTINCT nbr1.name) AS n1_neighbors,
            collect(DISTINCT nbr2.name) AS n2_neighbors
        """.format(l1=label1, l2=label2)
        
        # result, _ = db.cypher_query(query, {"n1": node1_name, "n2": node2_name})
        result = session.run(query, n1=node1_name, n2=node2_name).data()

    print("Neighbor Overlap Query Result:", result)

    if not result or not result[0][1] or not result[0][2]:
        return 0.0
    
    shared = result[0][0]
    n1_neighbors = result[0][1]
    n2_neighbors = result[0][2]
    
    # Jaccard overlap
    if not n1_neighbors or not n2_neighbors:
        return 0.0
    return len(shared) / len(set(n1_neighbors) | set(n2_neighbors))


def deduplication_score(node1, node2):
    """
    node1, node2: objects with attributes name, label, properties
    db: Neo4j connection context
    """

    fuzzy_score = fuzz.token_set_ratio(node1["name"], node2["name"]) / 100
    if fuzzy_score > 0.9:
        return 1.0

    emb1 = embedder.encode(node1["name"], convert_to_tensor=True)
    emb2 = embedder.encode(node2["name"], convert_to_tensor=True)

    # ---- 1. Embedding similarity ----
    # emb1 = model.encode(node1.name, normalize_embeddings=True)
    # emb2 = model.encode(node2.name, normalize_embeddings=True)
    name_sim = cosine_similarity(emb1, emb2)

    # ---- 2. Property similarity ----
    #prop_sim = compute_property_similarity(node1.properties, node2.properties)

    prop_sim = compute_property_similarity(json.loads(node1["properties"]), json.loads(node2["properties"]))

    # ---- 3. Structural (neighbor) similarity ----
    #neighbor_sim = compute_neighbor_overlap(node1.name, node2.name, node1.label, node2.label)
    neighbor_sim = compute_neighbor_overlap(node1["name"], node2["name"], node1["label"], node2["label"])

    # ---- 4. Weighted score ----
    final_score = 0.6 * name_sim + 0.3 * prop_sim + 0.1 * neighbor_sim

    return round(final_score, 3)

    # return {
    #     "name_sim": name_sim,
    #     "property_sim": prop_sim,
    #     "neighbor_sim": neighbor_sim,
    #     "final_score": final_score,
    # }

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
        score = deduplication_score(node_a, node_b)
        if score >= threshold:
            candidates.append({
                "node_a": node_a,
                "node_b": node_b,
                "score": score
            })
    return sorted(candidates, key=lambda x: x["score"], reverse=True)