from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import numpy as np
from itertools import combinations

class GraphDeduplicator:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def close(self):
        self.driver.close()

    def fetch_nodes(self):
        """Fetch all nodes with their label, name, and properties."""
        query = """
        MATCH (n)
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name, n.properties AS props
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [r.data() for r in result]

    def fetch_neighbors(self):
        """Fetch all node relationships for neighborhood similarity."""
        query = """
        MATCH (a)-[]->(b)
        RETURN id(a) AS src, collect(DISTINCT id(b)) AS neighbors
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {r["src"]: set(r["neighbors"]) for r in result}

    def compute_embedding_similarity(self, name1, name2):
        if not name1 or not name2:
            return 0.0
        emb1 = self.model.encode(name1, convert_to_tensor=True)
        emb2 = self.model.encode(name2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))

    def compute_property_similarity(self, props1, props2):
        """Compare dictionaries by overlapping key-value pairs."""
        if not props1 or not props2:
            return 0.0
        keys = set(props1.keys()) & set(props2.keys())
        matches = sum(1 for k in keys if props1[k] == props2[k])
        return matches / max(len(keys), 1)

    def compute_neighbor_similarity(self, n1, n2, neighbors):
        """Jaccard overlap of shared neighbors."""
        set1, set2 = neighbors.get(n1, set()), neighbors.get(n2, set())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def generate_merge_candidates(self, threshold=0.7):
        """Compare all pairs of nodes and return potential merge candidates."""
        nodes = self.fetch_nodes()
        neighbors = self.fetch_neighbors()

        candidates = []
        for n1, n2 in combinations(nodes, 2):
            # Skip if labels differ
            if n1["label"] != n2["label"]:
                continue

            emb_sim = self.compute_embedding_similarity(n1["name"], n2["name"])
            prop_sim = self.compute_property_similarity(n1["props"], n2["props"])
            neigh_sim = self.compute_neighbor_similarity(n1["id"], n2["id"], neighbors)

            final_score = 0.6 * emb_sim + 0.3 * prop_sim + 0.1 * neigh_sim

            if final_score >= threshold:
                candidates.append({
                    "node_1": n1,
                    "node_2": n2,
                    "embedding_sim": emb_sim,
                    "property_sim": prop_sim,
                    "neighbor_sim": neigh_sim,
                    "final_score": final_score
                })

        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
