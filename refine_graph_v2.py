#!/usr/bin/env python3
"""
Neo4j Graph Deduplication Script
--------------------------------
- Extracts nodes from Neo4j
- Compares based on name embeddings + property rules
- Merges duplicates using APOC (or dry-run mode)
"""

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import torch
import json

# -----------------------------
# CONFIG
# -----------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"

NODE_LABEL = "Person"   # change to the label you want
NAME_THRESHOLD = 0.85   # cosine similarity threshold for names
PROP_THRESHOLD = 0.50   # proportion of matching properties
DRY_RUN = True          # ⚠️ True = only print duplicates, False = actually merge


# -----------------------------
# Embedding Model
# -----------------------------
model = SentenceTransformer("intfloat/e5-base-v2")


# -----------------------------
# Similarity Functions
# -----------------------------
def property_similarity(props_a: dict, props_b: dict):
    """Compare property dictionaries directly (rule-based)."""
    if not props_a or not props_b:
        return 1.0  # no conflict if one is empty

    overlap = 0
    total = 0
    for key in set(props_a.keys()).union(props_b.keys()):
        total += 1
        if key in props_a and key in props_b:
            if props_a[key] == props_b[key]:
                overlap += 1
            else:
                return 0.0  # hard conflict → block merge
    return overlap / total if total > 0 else 1.0


def is_duplicate(name_sim, props_a, props_b,
                 name_threshold=NAME_THRESHOLD,
                 prop_threshold=PROP_THRESHOLD):
    """Check if two nodes are duplicates."""
    if name_sim < name_threshold:
        return False
    prop_sim = property_similarity(props_a, props_b)
    print("prop_sim:", prop_sim)
    return prop_sim >= prop_threshold


# -----------------------------
# Neo4j Utilities
# -----------------------------
def get_nodes(tx, label):
    query = f"MATCH (n:{label}) RETURN id(n) as id, n.name as name, properties(n) as props"
    result = tx.run(query)
    return [{"id": record["id"], "name": record["name"], "properties": record["props"]} for record in result]


def merge_nodes(tx, id1, id2):
    query = """
    MATCH (a), (b)
    WHERE id(a) = $id1 AND id(b) = $id2
    CALL apoc.refactor.mergeNodes([a,b], {properties:'combine'}) YIELD node
    RETURN id(node) as id
    """
    tx.run(query, id1=id1, id2=id2)

# -----------------------------
# Main Deduplication Pipeline
# -----------------------------
def deduplicate_graph(label):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        print(f"Fetching nodes with label '{label}'...")
        nodes = session.read_transaction(get_nodes, label)

    if not nodes:
        print("No nodes found.")
        return

    print("nodes:", nodes)
    print(f"Found {len(nodes)} nodes. Computing embeddings...")
    names = [n["name"] for n in nodes]
    embeddings = model.encode(names, convert_to_tensor=True, show_progress_bar=True)
    print("embeddings:", embeddings)
    print("Comparing nodes...")
    pairs_to_merge = []
    sim_matrix = util.cos_sim(embeddings, embeddings)
    print("sim_matrix:", sim_matrix)
    for i, node_a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node_b = nodes[j]
            name_sim = sim_matrix[i][j].item()
            print("name_sim:", name_sim)
            print("node_a:", node_a["name"], "node_b:", node_b["name"])
            print("node_a:", node_a["properties"], "node_b:", node_b["properties"])
            if is_duplicate(name_sim, json.loads(node_a["properties"]['properties']), json.loads(node_b["properties"]['properties'])):
                pairs_to_merge.append((node_a, node_b, name_sim))

    print(f"Found {len(pairs_to_merge)} duplicate pairs.")

    if DRY_RUN:
        print("\n⚠️ DRY RUN MODE: No changes will be made in Neo4j")
        for a, b, sim in pairs_to_merge:
            print(f" - {a['name']} (id={a['id']})  <->  {b['name']} (id={b['id']}) "
                  f"(sim={sim:.2f})")
    else:
        print("Merging duplicates in Neo4j...")
        with driver.session() as session:
            for a, b, sim in pairs_to_merge:
                session.write_transaction(merge_nodes, a["id"], b["id"])
                print(f"Merged {a['name']} (id={a['id']}) with {b['name']} (id={b['id']})")
        print("Deduplication complete ✅")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    deduplicate_graph()
