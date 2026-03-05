import faiss
import numpy as np
from sklearn.preprocessing import normalize
from collections import defaultdict
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
from knowledge_graph_maker import Edge, Node
from transformers import pipeline

class ClusterER:
    def __init__(self, pred2idx: list=None, sim_threshold: float=0.65, embedding_model: str="all-mpnet-base-v2"):
        """
        pred2idx: mapping predicate -> index for relational signatures
        sim_threshold: threshold to merge with existing cluster
        """
        self.model = SentenceTransformer(embedding_model)
        self.ner_pipeline = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
        )
        self.sim_threshold = sim_threshold

        # FAISS for cluster embeddings
        self.dim = 768  # mpnet embedding size
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.int2cluster = {}  # mapping FAISS row -> cluster_id
        self.cluster_counter = 1

        # Clusters
        self.clusters = {}  # cluster_id -> cluster dict

        # Relational signature
        self.pred2idx = pred2idx or {}
        self.num_predicates = len(self.pred2idx)

    # ----------------------
    # Helper: String similarity
    # ----------------------
    def _string_sim(self, a: str, b: str)-> float: 
        return fuzz.token_sort_ratio(a, b) / 100

    # ----------------------
    # Helper: Type compatibility
    # ----------------------
    # def _predict_type(self, entity_text, predicate=None):
    #     # Dummy placeholder: override with NER model if desired
    #     return "UNKNOWN"
    
    def _predict_type(self, entity_text: str, predicate=None) -> str:
        text = f"{entity_text} {predicate}" if predicate else entity_text

        try:
            entities = self.ner_pipeline(text)
        except Exception:
            return "UNKNOWN"

        for ent in entities:
            if ent["word"].lower() in entity_text.lower():
                label = ent["entity_group"]
                return {
                    "PER": "PERSON",
                    "ORG": "ORGANIZATION",
                    "LOC": "LOCATION",
                    "MISC": "MISC"
                }.get(label, "UNKNOWN")

        return "UNKNOWN"

    def _types_compatible(self, type1: str, type2: str) -> bool:
        if type1 == "UNKNOWN" or type2 == "UNKNOWN":
            return True
        return type1 == type2

    # ----------------------
    # FAISS utilities
    # ----------------------
    def _add_to_faiss(self, cluster_id: str, emb: np.ndarray):
        row = self.faiss_index.ntotal
        self.faiss_index.add(emb)
        self.int2cluster[row] = cluster_id

    def _semantic_blocking(self, entity_text: str, top_k: int=10):
        """Retrieve candidate clusters via FAISS"""
        if self.faiss_index.ntotal == 0:
            return []

        query_emb = self.model.encode(entity_text)
        query_emb = np.asarray(query_emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_emb)

        scores, indices = self.faiss_index.search(query_emb, top_k)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < self.sim_threshold:
                continue
            candidates.append(self.int2cluster[idx])
        return candidates

    # ----------------------
    # Relational signatures
    # ----------------------
    def _compute_candidate_relvec(self, predicate: str=None, other_entity_id: str=None):
        """Build candidate relational vector for current triple"""
        vec = np.zeros((1, self.num_predicates * 2), dtype=np.float32)
        if predicate in self.pred2idx:
            p_idx = self.pred2idx[predicate]
            vec[0, p_idx] += 1  # subject role
            if other_entity_id is not None:
                vec[0, p_idx + self.num_predicates] += 1  # object role
        return vec

    def _cluster_relational_similarity(self, cluster_id: str, candidate_vec: np.ndarray=None):
        cluster_vec = self.clusters[cluster_id]["rel_signature"]
        if candidate_vec is None:
            candidate_vec = np.zeros_like(cluster_vec)

        # cosine similarity
        c_norm = cluster_vec / (np.linalg.norm(cluster_vec) + 1e-10)
        e_norm = candidate_vec / (np.linalg.norm(candidate_vec) + 1e-10)
        return float(np.dot(c_norm, e_norm.T))

    # ----------------------
    # Cluster management
    # ----------------------
    def _create_new_cluster(self, entity_text: str, predicate=None) -> tuple[str, str]:
        cluster_id = f"C{self.cluster_counter}"
        self.cluster_counter += 1

        emb = self.model.encode(entity_text)
        emb = np.asarray(emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(emb)

        rel_vec = self._compute_candidate_relvec(predicate)

        self.clusters[cluster_id] = {
            "members": {entity_text},
            "centroid": emb,
            "type": self._predict_type(entity_text, predicate),
            "neighbors": set(),
            "rel_signature": rel_vec
        }

        self._add_to_faiss(cluster_id, emb)
        return cluster_id, "INSERT"

    def _update_cluster(self, cluster_id: str, new_entity_text: str, predicate=None):
        cluster = self.clusters[cluster_id]

        new_emb = self.model.encode(new_entity_text)
        new_emb = np.asarray(new_emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(new_emb)

        # Update centroid
        cluster["centroid"] = normalize(cluster["centroid"] + new_emb, norm="l2")
        cluster["members"].add(new_entity_text)

        # Update relational signature
        if predicate:
            candidate_vec = self._compute_candidate_relvec(predicate)
            cluster["rel_signature"] += candidate_vec

        # Rebuild FAISS for simplicity
        self._rebuild_faiss()

    def _rebuild_faiss(self):
        self.faiss_index.reset()
        self.int2cluster = {}
        for cid, c in self.clusters.items():
            self._add_to_faiss(cid, c["centroid"])

    # ----------------------
    # Cluster-level similarity
    # ----------------------
    def _cluster_similarity(self, entity_text: str, cluster_id: str, predicate=None, other_entity_id=None):
        cluster = self.clusters[cluster_id]

        # string similarity
        s_sim = max(self._string_sim(entity_text, m) for m in cluster["members"])

        # embedding similarity
        context = f"{entity_text} in relation {predicate}" if predicate else entity_text
        emb_query = self.model.encode(context)
        e_sim = util.cos_sim(emb_query, cluster["centroid"]).item()

        # type similarity
        new_type = self._predict_type(entity_text, predicate)
        p_sim = 1.0 if self._types_compatible(new_type, cluster["type"]) else 0.0

        # relational similarity
        candidate_relvec = self._compute_candidate_relvec(predicate, other_entity_id)
        r_sim = self._cluster_relational_similarity(cluster_id, candidate_relvec)

        # composite
        return 0.25 * s_sim + 0.40 * e_sim + 0.10 * p_sim + 0.25 * r_sim

    # ----------------------
    # Resolve entity
    # ----------------------
    def resolve_entity(self, entity_text: str, predicate=None, other_entity_id=None):
        candidates = self._semantic_blocking(entity_text)
        best_score = 0
        best_cluster = None

        for cluster_id in candidates:
            score = self._cluster_similarity(entity_text, cluster_id, predicate, other_entity_id)
            if score > best_score:
                best_score = score
                best_cluster = cluster_id

        if best_cluster and best_score >= self.sim_threshold:
            self._update_cluster(best_cluster, entity_text, predicate)
            return best_cluster, "MERGE"

        return self._create_new_cluster(entity_text, predicate)

    # ----------------------
    # Get entity -> cluster mapping
    # ----------------------
    def get_entity_mapping(self):
        mapping = {}
        for cid, cluster in self.clusters.items():
            for m in cluster["members"]:
                mapping[m] = cid
        return mapping

    # ----------------------
    # Evaluation functions
    # ----------------------
    @staticmethod
    def pairwise_links(items: list) -> set:
        if len(items) == 0:
            return set()
        if len(items) == 1:
            return {frozenset(items)}
        return set(frozenset([a, b]) for a, b in combinations(items, 2))

    @staticmethod
    def build_gold_entity_map(gold_clusters: defaultdict[str, set]) -> defaultdict[str, set]:
        gold_entity_map = defaultdict(set)
        for gold_cluster, entities in gold_clusters.items():
            for e in entities:
                gold_entity_map[e].add(gold_cluster)
        return gold_entity_map

    @staticmethod
    def evaluate_cluster_alignment(gold_clusters: defaultdict[str, set], predicted_clusters: defaultdict[str, set]):
        gold_entity_map = ClusterER.build_gold_entity_map(gold_clusters)
        for pred_cluster_id, pred_entities in predicted_clusters.items():
            matched_gold = defaultdict(set)
            for e in pred_entities:
                golds = gold_entity_map.get(e, set())
                if not golds:
                    matched_gold["UNSEEN"].add(e)
                else:
                    for g in golds:
                        matched_gold[g].add(e)

            if len(matched_gold) == 1:
                print(f"Pure match with gold cluster: {next(iter(matched_gold))}")
            else:
                print("Mixed cluster matches:")
                for g, ents in matched_gold.items():
                    print(f"  - {g}: {ents}")

    def evaluate(self, toy_dataset: list[dict]):
        # build gold clusters
        gold_clusters = defaultdict(set)
        for t in toy_dataset:
            gold_clusters[t["subject_gold"]].add(t["subject"])
            gold_clusters[t["object_gold"]].add(t["object"])

        # predicted clusters
        pred_clusters = defaultdict(set)
        mapping = self.get_entity_mapping()
        for mention, cid in mapping.items():
            pred_clusters[cid].add(mention)

        # pairwise metrics
        gold_links = set()
        for entities in gold_clusters.values():
            gold_links.update(self.pairwise_links(entities))
        pred_links = set()
        for entities in pred_clusters.values():
            pred_links.update(self.pairwise_links(entities))

        tp = len(gold_links & pred_links)
        fp = len(pred_links - gold_links)
        fn = len(gold_links - pred_links)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        print("\n=== Evaluation Results ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")

        print("\n=== Cluster Alignment ===")
        self.evaluate_cluster_alignment(gold_clusters, pred_clusters)
        return precision, recall, f1
    
    def deduplicate_edges(self, list_of_edges: list[Edge], merge_metadata=True) -> tuple[list[Edge], int]:
        """
        Remove or merge duplicate edges after entity resolution.

        Returns:
            unique_edges: list of canonicalized edges
            duplicate_count: number of removed edges
        """

        canonical_edges = {}
        duplicate_count = 0

        for edge in list_of_edges:
            s_text = edge.node_1.name
            o_text = edge.node_2.name
            predicate = edge.relationship

            # Resolve to cluster IDs
            s_cluster = self.get_entity_mapping().get(s_text)
            o_cluster = self.get_entity_mapping().get(o_text)

            # If resolution failed, skip
            if s_cluster is None or o_cluster is None:
                continue

            # Canonical triple key
            key = (s_cluster, predicate, o_cluster)

            if key not in canonical_edges:
                canonical_edges[key] = {
                    "subject_cluster": s_cluster,
                    "predicate": predicate,
                    "object_cluster": o_cluster,
                    "mentions": [(s_text, o_text)],
                    "metadata": [edge.metadata]
                }
            else:
                duplicate_count += 1
                canonical_edges[key]["mentions"].append((s_text, o_text))

                if merge_metadata:
                    canonical_edges[key]["metadata"].append(edge.metadata)

        return list(canonical_edges.values()), duplicate_count
    
    def process_edges(self, list_of_edges: list[Edge]) -> list[Edge]:
        for edge in list_of_edges:

            subj = edge.node_1.name
            obj = edge.node_2.name
            pred = edge.relationship

            subj_cluster, _ = self.resolve_entity(subj, predicate=pred)
            obj_cluster, _ = self.resolve_entity(obj, predicate=pred,
                                                other_entity_id=subj_cluster)

        return self.deduplicate_edges_to_graph(list_of_edges)
        
    def deduplicate_edges_to_graph(self, list_of_edges: list[Edge]) -> list[Edge]:
        """
        Deduplicate edges but reconstruct them using canonical mention names
        instead of cluster IDs.
        """

        entity_map = self.get_entity_mapping()

        # Step 1: Build cluster -> canonical mention name mapping
        cluster_to_name = {}
        cluster_to_label = {}

        for cluster_id, cluster_data in self.clusters.items():
            # Choose first inserted mention as canonical
            canonical_name = next(iter(cluster_data["members"]))
            cluster_to_name[cluster_id] = canonical_name

        canonical_edges = {}
        new_order = 0

        for edge in list_of_edges:
            s_text = edge.node_1.name
            o_text = edge.node_2.name
            predicate = edge.relationship

            s_cluster = entity_map.get(s_text)
            o_cluster = entity_map.get(o_text)

            if s_cluster is None or o_cluster is None:
                continue

            key = (s_cluster, predicate, o_cluster)

            if key not in canonical_edges:
                canonical_edges[key] = {
                    "subject_cluster": s_cluster,
                    "object_cluster": o_cluster,
                    "predicate": predicate,
                    "subject_label": edge.node_1.label,
                    "object_label": edge.node_2.label,
                    "metadata": [edge.metadata],
                    "order": new_order
                }
                new_order += 1
            else:
                canonical_edges[key]["metadata"].append(edge.metadata)

        # Step 2: Reconstruct graph using canonical mention names
        deduplicated_edges = []

        for data in canonical_edges.values():

            subject_node = Node(
                label=data["subject_label"],
                name=cluster_to_name[data["subject_cluster"]],
                properties={}
            )

            object_node = Node(
                label=data["object_label"],
                name=cluster_to_name[data["object_cluster"]],
                properties={}
            )

            merged_metadata = {
                "merged_from": data["metadata"]
            }

            deduplicated_edges.append(
                Edge(
                    node_1=subject_node,
                    node_2=object_node,
                    relationship=data["predicate"],
                    metadata=merged_metadata,
                    order=data["order"]
                )
            )

        return deduplicated_edges
    
def clean_edges_for_neo4j(edges: list[Edge]) -> list[Edge]:
    """
    Take a list of Edge objects and return a cleaned version suitable for Neo4j insertion.
    - Replaces spaces in labels with underscores.
    - Escapes quotes in names.
    - Keeps metadata intact but ensures all values are strings and quotes are escaped.
    Returns a new list of Edge objects.
    """
    from copy import deepcopy
    cleaned_edges = []

    for edge in edges:
        # Deepcopy to avoid modifying original objects
        new_edge = deepcopy(edge)

        # Clean node labels (replace spaces with underscores)
        new_edge.node_1.label = new_edge.node_1.label.replace(" ", "_")
        new_edge.node_2.label = new_edge.node_2.label.replace(" ", "_")

        # Clean node names (escape double quotes)
        new_edge.node_1.name = new_edge.node_1.name.replace('"', '\\"')
        new_edge.node_2.name = new_edge.node_2.name.replace('"', '\\"')

        # Convert all metadata values to strings and escape quotes
        new_metadata = {}
        for k, v in new_edge.metadata.items():
            new_metadata[k] = str(v).replace('"', '\\"')
        new_edge.metadata = new_metadata

        # Node properties: ensure all values are strings and escape quotes
        for node in [new_edge.node_1, new_edge.node_2]:
            new_props = {}
            for k, v in node.properties.items():
                new_props[k] = str(v).replace('"', '\\"')
            node.properties = new_props

        cleaned_edges.append(new_edge)

    return cleaned_edges